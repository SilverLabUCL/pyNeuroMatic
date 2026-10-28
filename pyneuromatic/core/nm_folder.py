# -*- coding: utf-8 -*-
"""
[Module description].

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
import dataclasses
import datetime
from typing import Any, TYPE_CHECKING
import h5py

from pyneuromatic.core.nm_data import NMData, NMDataContainer
from pyneuromatic.core.nm_dataseries import NMDataSeries, NMDataSeriesContainer
from pyneuromatic.core.nm_notes import NMNotes
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
import pyneuromatic.core.nm_command_history as nmch
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_utilities as nmu

if TYPE_CHECKING:
    # nm_tool_folder (pyneuromatic.tools) imports NMFolder for type hints,
    # so importing it at module level here would create a core<->tools
    # circular import; instantiation below uses a lazy, in-function import
    # instead. See issue #325.
    from pyneuromatic.tools.nm_tool_folder import NMToolFolderContainer


"""
NM class tree:

NMManager (NMObject, root)
    NMFolderContainer
        NMFolder (folder0, folder1...)
                NMDataContainer
                    NMData (recordA0, recordA1... avgA0, avgB0)
                NMDataSeriesContainer
                    NMDataSeries (record, avg...)
                        NMChannelContainer
                            NMChannel (A, B, C...)
                        NMEpochContainer
                            NMEpoch (E0, E1, E2...)
"""


@dataclasses.dataclass(frozen=True)
class NMDataSeriesScan:
    """Result of :meth:`NMFolder.scan_dataseries` — what would be created.

    Attributes:
        prefix:   Detected dataseries name (e.g. ``"Record"``).
        channels: Sorted list of channel characters (e.g. ``["A", "B"]``).
        epochs:   Sorted list of epoch numbers (e.g. ``[0, 1, 2]``).
        n_data:   Total number of matching data items.
    """
    prefix: str
    channels: list
    epochs: list
    n_data: int


class NMFolder(NMObject):
    """
    NM Data Folder class
    """

    # Extend NMObject's special attrs with NMFolder's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMFolder__data_container",
        "_NMFolder__dataseries_container",
        "_NMFolder__toolfolder_container",
        "_NMFolder__toolresults",
        "_NMFolder__metadata",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMFolder0",
    ) -> None:
        super().__init__(parent=parent, name=name)

        # Lazy import: see the TYPE_CHECKING import note above (issue #325).
        from pyneuromatic.tools.nm_tool_folder import NMToolFolderContainer

        self.__data_container: NMDataContainer = NMDataContainer(parent=self)
        self.__dataseries_container: NMDataSeriesContainer = NMDataSeriesContainer(parent=self)
        self.__toolfolder_container: NMToolFolderContainer = NMToolFolderContainer(parent=self)
        self.__toolresults: dict[str, object] = {}  # tool results saved to dict
        self.__notes = NMNotes()
        self.__metadata: dict[str, dict] = {}  # nested dict by source folder

    # override
    def __eq__(
        self,
        other: object
    ) -> bool:
        if not isinstance(other, NMFolder):
            return NotImplemented
        if not super().__eq__(other):
            return False
        if self.__data_container != other.__data_container:
            return False
        if self.__dataseries_container != other.__dataseries_container:
            return False
        if self.__toolfolder_container != other.__toolfolder_container:
            return False
        if self.__toolresults != other.__toolresults:
            return False
        if self.__notes != other.__notes:
            return False
        if self.__metadata != other.__metadata:
            return False
        return True

    def __deepcopy__(self, memo: dict) -> NMFolder:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMFolder by bypassing __init__ and directly
        setting attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMFolder
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Use the class attribute for special attrs (includes NMObject's attrs)
        special_attrs = cls._DEEPCOPY_SPECIAL_ATTRS

        # Deep copy all attributes that aren't special
        for attr, value in self.__dict__.items():
            if attr not in special_attrs:
                setattr(result, attr, copy.deepcopy(value, memo))

        # Set NMObject's attributes with custom handling
        result._NMObject__created = datetime.datetime.now().isoformat(" ", "seconds")
        result._NMObject__parent = self._NMObject__parent
        result._NMObject__name = self._NMObject__name
        result._container = None
        result._NMObject__copy_of = self

        # Now handle NMFolder's special attributes

        # __data_container: deep copy and update parent
        result._NMFolder__data_container = copy.deepcopy(
            self._NMFolder__data_container, memo
        )
        result._NMFolder__data_container._parent = result

        # __dataseries_container: deep copy and update parent
        result._NMFolder__dataseries_container = copy.deepcopy(
            self._NMFolder__dataseries_container, memo
        )
        result._NMFolder__dataseries_container._parent = result

        # __toolfolder_container: deep copy and update parent
        result._NMFolder__toolfolder_container = copy.deepcopy(
            self._NMFolder__toolfolder_container, memo
        )
        result._NMFolder__toolfolder_container._parent = result

        # __toolresults: deep copy the dict
        result._NMFolder__toolresults = copy.deepcopy(
            self._NMFolder__toolresults, memo
        )

        # __metadata: deep copy the nested dict
        result._NMFolder__metadata = copy.deepcopy(
            self._NMFolder__metadata, memo
        )

        # Copied arrays whose x array was copied later point to that copy
        from pyneuromatic.core.nm_data import XARRAY_LINKS
        for data, source in memo.pop(XARRAY_LINKS, []):
            data._NMData__xarray_source = memo.get(id(source), source)

        return result

    # override
    @property
    def content(self) -> dict[str, str]:
        k = super().content
        k.update(self.__data_container.content)
        k.update(self.__dataseries_container.content)
        k.update(self.__toolfolder_container.content)
        return k

    @property
    def data(self) -> NMDataContainer:
        return self.__data_container

    @property
    def dataseries(self) -> NMDataSeriesContainer:
        return self.__dataseries_container

    @property
    def toolfolders(self) -> NMToolFolderContainer:
        return self.__toolfolder_container

    @property
    def toolresults(self) -> dict[str, object]:
        return self.__toolresults

    def toolresults_save(
        self,
        tool: str,
        results: Any,
        quiet: bool = nmc.QUIET,
    ) -> int:
        if not isinstance(tool, str):
            e = nmu.type_error_str(tool, "tool", "string")
            raise TypeError(e)

        if tool not in self.__toolresults:
            self.__toolresults[tool] = []
        entry = {"date": str(datetime.datetime.now()), "results": results}
        self.__toolresults[tool].append(entry)
        idx = len(self.__toolresults[tool]) - 1
        nmh.history(
            "saved %s results to %s[%d]" % (tool, self.path_str, idx),
            quiet=quiet,
        )
        return idx

    def toolresults_clear(
        self,
        tool: str | None = None,
        idx: int | None = None,
        quiet: bool = nmc.QUIET,
    ) -> None:
        if tool is None:
            self.__toolresults.clear()
            nmh.history("cleared all toolresults", quiet=quiet)
            return None
        if not isinstance(tool, str):
            raise TypeError(nmu.type_error_str(tool, "tool", "string"))
        if tool not in self.__toolresults:
            raise KeyError("tool not found in toolresults: %s" % tool)
        if idx is None:
            del self.__toolresults[tool]
            nmh.history(
                "cleared %s toolresults" % tool,
                quiet=quiet,
            )
            return None
        if not isinstance(idx, int):
            raise TypeError(nmu.type_error_str(idx, "idx", "integer"))
        entries = self.__toolresults[tool]
        if idx < 0 or idx >= len(entries):
            raise IndexError(
                "idx %d out of range for %s toolresults (len=%d)"
                % (idx, tool, len(entries))
            )
        entries.pop(idx)
        if len(entries) == 0:
            del self.__toolresults[tool]
        nmh.history(
            "cleared %s toolresults[%d]" % (tool, idx),
            quiet=quiet,
        )
        return None

    @property
    def notes(self) -> NMNotes:
        """Return notes for this folder."""
        return self.__notes

    # Metadata - structured key-value data from imported files

    @property
    def metadata(self) -> dict[str, dict]:
        """Return metadata dict for this folder.

        Metadata is a nested dict preserving the folder structure from the
        source file. For PXP files, keys are Igor folder names mapping to
        dicts of variables/strings found in that folder.

        Example:
            >>> folder.metadata["root"]["AcqMode"]
            'episodic'
            >>> folder.metadata["Notes"]["H_Name"]
            'Jason Rothman'
        """
        return self.__metadata

    # ------------------------------------------------------------------
    # Helpers for dataseries methods
    # ------------------------------------------------------------------

    def detect_data_prefixes(self) -> list[str]:
        """Detect unique data prefixes in this folder's data container.

        Scans all data names and parses them to find NeuroMatic naming patterns
        ({prefix}{channel}{epoch}). Returns a sorted list of unique prefixes.

        Returns:
            List of unique prefix strings found (e.g., ["Record", "avg"]).

        Example:
            If folder contains: RecordA0, RecordA1, RecordB0, avgA0, avgB0
            Returns: ["Record", "avg"]
        """
        prefixes: set[str] = set()
        for name in self.data.keys():
            parsed = nmu.parse_data_name(name)
            if parsed is not None:
                prefix, _, _ = parsed
                prefixes.add(prefix)
        return sorted(prefixes)

    def _get_dataseries(self, prefix: str):
        """Return the NMDataSeries for *prefix*, or raise ValueError."""
        ds = self.dataseries.get(prefix)
        if ds is None:
            raise ValueError("dataseries %r not found" % prefix)
        return ds

    def _get_unlocked_dataseries(self, prefix: str):
        """Return the NMDataSeries for *prefix*; raise if it is locked."""
        ds = self._get_dataseries(prefix)
        if ds.locked:
            raise RuntimeError("dataseries %r is locked" % ds.name)
        return ds

    def rename_dataseries(
        self,
        prefix: str,
        new_prefix: str,
        quiet: bool = nmc.QUIET,
    ) -> NMDataSeries:
        """Rename a dataseries and its data arrays (RecordA0 -> CellA0, ...).

        Raises:
            ValueError: If *prefix* is not a dataseries, *new_prefix* is not a
                valid name or is already a dataseries, or a new data name is
                already taken.
            RuntimeError: If the dataseries is locked.
        """
        ds = self._get_unlocked_dataseries(prefix)
        if not isinstance(new_prefix, str) or not new_prefix or not nmu.name_ok(new_prefix):
            raise ValueError("new prefix: %r" % new_prefix)
        if self.dataseries._getkey(new_prefix) is not None:
            raise ValueError("dataseries %r already exists" % new_prefix)
        members: list[NMData] = []
        for channel in ds.channels.values():
            for data in channel.data:
                if data not in members:
                    members.append(data)
        renames = {
            data.name: new_prefix + data.name[len(ds.name):]
            for data in members
            if data.name.lower().startswith(ds.name.lower())
        }
        member_keys = {name.lower() for name in renames}
        for new_name in renames.values():
            key = self.data._getkey(new_name)
            if key is not None and key.lower() not in member_keys:
                raise ValueError("data %r already exists" % new_name)
        # Rename the arrays directly: NMDataContainer.rename() would unlink
        # them from the series, which keeps the same members here
        for old_name, new_name in renames.items():
            NMObjectContainer.rename(self.data, old_name, new_name, quiet=True)
        old = ds.name
        self.dataseries._rename_locked(old, new_prefix, quiet=True)
        nmh.history(
            "renamed dataseries '%s' as '%s' (%d data)" % (old, new_prefix, len(renames)),
            path=self.path_str,
            quiet=quiet,
        )
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        nmch.add_nm_command('%s.rename_dataseries(%r, %r)' % (_p, old, new_prefix))
        return ds

    def renumber_epochs_plan(self, prefix: str, remove_empty: bool = False) -> tuple:
        """What :meth:`renumber_dataseries_epochs` would do, without changing
        anything: ({old epoch: new epoch}, {old array: new array}, [empty
        epochs removed]), only those that change. With *remove_empty*,
        epochs with no arrays (e.g. their arrays were deleted in the Data
        list) are removed first.

        Raises:
            RuntimeError: If the series is locked.
            ValueError: If a new array name is taken by an array outside the
                series.
        """
        ds = self._get_unlocked_dataseries(prefix)
        epoch_prefix = ds.epochs.auto_name_prefix
        empty = [e.name for e in ds.epochs.values() if not e.data] if remove_empty else []
        epochs = {}
        kept = [name for name in ds.epochs.keys() if name not in empty]
        for index, name in enumerate(kept):
            new = "%s%d" % (epoch_prefix, index)
            if new != name:
                epochs[name] = new
        arrays = {}
        for old, new in epochs.items():
            old_num, new_num = old[len(epoch_prefix):], new[len(epoch_prefix):]
            for channel in ds.channels.values():
                for data in ds.epochs[old].data:
                    expected = "%s%s%s" % (ds.name, channel.name, old_num)
                    if data.name == expected and any(d is data for d in channel.data):
                        arrays[data.name] = "%s%s%s" % (ds.name, channel.name, new_num)
        renamed = {name.lower() for name in arrays}
        for new in arrays.values():
            key = self.data._getkey(new)
            if key is not None and key.lower() not in renamed:
                raise ValueError(
                    "data '%s' already exists (not part of data series '%s')"
                    % (new, ds.name))
        return epochs, arrays, empty

    def renumber_dataseries_epochs(self, prefix: str, remove_empty: bool = False,
                                   quiet: bool = nmc.QUIET) -> dict:
        """Number the epochs of data series *prefix* E0, E1, ... in order,
        closing gaps (e.g. E0, E2, E5 -> E0, E1, E2), and rename their arrays
        to match (RecordA2 -> RecordA1). With *remove_empty*, epochs with no
        arrays are removed first. Epoch sets and groups follow; the series'
        note and each renamed array's note record the old names.

        Returns:
            {old epoch name: new epoch name} for the epochs renamed.

        Raises:
            RuntimeError: If the series is locked.
            ValueError: If a new array name is taken outside the series.
        """
        epochs, arrays, empty = self.renumber_epochs_plan(prefix, remove_empty)
        ds = self.dataseries[prefix]
        if empty:
            self.remove_dataseries_epoch(prefix, empty, quiet=True)
            ds.notes.add("removed empty epochs: %s" % ", ".join(empty))
        if not epochs:
            return {}
        groups = ds.epochs.groups
        # Two passes (via temporary names), so renames never collide; the
        # arrays are renamed directly, keeping them in the series
        temp = {old: "%s_renumber_%d" % (old, i) for i, old in enumerate(arrays)}
        for old, tmp in temp.items():
            NMObjectContainer.rename(self.data, old, tmp, quiet=True)
        for old, tmp in temp.items():
            NMObjectContainer.rename(self.data, tmp, arrays[old], quiet=True)
            self.data[arrays[old]].notes.add(
                "renamed from '%s' (Renumber Epochs)" % old)
        temp_epochs = {old: "%s_renumber" % old for old in epochs}
        for old, tmp in temp_epochs.items():
            ds.epochs._rename_locked(old, tmp, quiet=True)  # sets follow
            groups.rename_item(old, tmp)
        epoch_prefix = ds.epochs.auto_name_prefix
        for old, tmp in temp_epochs.items():
            ds.epochs._rename_locked(tmp, epochs[old], quiet=True)
            groups.rename_item(tmp, epochs[old])
            ds.epochs[epochs[old]].number = int(epochs[old][len(epoch_prefix):])
        changes = ", ".join("%s → %s" % item for item in epochs.items())
        ds.notes.add("renumbered epochs: %s" % changes)
        nmh.history(
            "renumbered epochs of dataseries '%s': %s (%d arrays renamed)"
            % (prefix, changes, len(arrays)),
            path=self.path_str, quiet=quiet,
        )
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        nmch.add_nm_command('%s.renumber_dataseries_epochs(%r, remove_empty=%r)'
                            % (_p, prefix, remove_empty))
        return epochs

    def _resolve_channel_names(self, ds, channel) -> list[str]:
        """Resolve *channel* (str | int | list | range) to a list of channel chars.

        Raises TypeError for unsupported types, ValueError for chars not in *ds*.
        """
        if isinstance(channel, range):
            items = list(channel)
        elif isinstance(channel, list):
            items = channel
        else:
            items = [channel]

        names: list[str] = []
        for item in items:
            if isinstance(item, bool):
                raise TypeError(
                    "channel must be str or int, got bool"
                )
            if isinstance(item, int):
                ch_name = nmu.channel_char(item)
                if not ch_name:
                    raise ValueError("invalid channel number: %d" % item)
            elif isinstance(item, str):
                ch_name = item.upper()
            else:
                raise TypeError(
                    "channel must be str or int, got %s" % type(item).__name__
                )
            if ds.channels.get(ch_name) is None:
                raise ValueError(
                    "channel %r not found in dataseries %r" % (ch_name, ds.name)
                )
            names.append(ch_name)
        return names

    def _resolve_epoch_names(self, ds, epoch) -> list[str]:
        """Resolve *epoch* (str | int | list | range) to a list of epoch names.

        Epoch names are stored as 'E0', 'E1', etc.
        Raises TypeError for unsupported types, ValueError for names not in *ds*.
        """
        if isinstance(epoch, range):
            items = list(epoch)
        elif isinstance(epoch, list):
            items = epoch
        else:
            items = [epoch]

        names: list[str] = []
        for item in items:
            if isinstance(item, bool):
                raise TypeError(
                    "epoch must be str or int, got bool"
                )
            if isinstance(item, int):
                ep_name = "E%d" % item
            elif isinstance(item, str):
                ep_name = item
            else:
                raise TypeError(
                    "epoch must be str or int, got %s" % type(item).__name__
                )
            if ds.epochs.get(ep_name) is None:
                raise ValueError(
                    "epoch %r not found in dataseries %r" % (ep_name, ds.name)
                )
            names.append(ep_name)
        return names

    # ------------------------------------------------------------------
    # Dataseries methods (build, new, sync, copy, remove)
    # ------------------------------------------------------------------

    @property
    def _nm_cmd_path(self) -> str:
        """Command-history path for this folder, derived from parent.

        Returns e.g. ``'folders["TestFolder"]'`` when the parent is an
        NMManager, so that ``nmch.add_nm_command(_nm_cmd_path + ".method()")``
        produces ``nm.folders["TestFolder"].method()``.
        """
        from pyneuromatic.core.nm_manager import NMManager
        if isinstance(self._parent, NMManager):
            return 'folders["%s"]' % self.name
        return self.name

    def build_dataseries(
        self,
        prefix: str,
        matches: dict[tuple[str, int], NMData],
        select: bool = False,
        quiet: bool = nmc.QUIET,
    ) -> NMDataSeries | None:
        """Build or extend an NMDataSeries from an explicit matches mapping.

        Creates the dataseries if it does not yet exist, then creates any
        channels and epochs not already present, and links each NMData to
        its channel and epoch.  Already-linked data is silently skipped so
        this method is safe to call on an existing dataseries.

        This is the core implementation shared by :meth:`sync_dataseries`
        (which first scans the folder to build *matches*) and
        :meth:`copy_dataseries` (which builds *matches* directly).

        Args:
            prefix: Exact dataseries name to create or update.
            matches: Mapping of ``(channel_char, epoch_num) -> NMData``.
                Channel chars must be single uppercase letters (e.g. ``"A"``);
                epoch numbers are zero-based integers.
            select: Whether to select the dataseries after creation/update.
            quiet: Suppress history messages (caller is responsible for
                logging a meaningful summary).

        Returns:
            The NMDataSeries (new or updated), or ``None`` on failure.
        """
        if not matches:
            return None

        # Get or create the dataseries
        is_new = prefix not in self.dataseries
        if is_new:
            ds = self.dataseries.new(name=prefix, select=select, quiet=True)
            if ds is None:
                return None
            channel_map: dict[str, object] = {}
            epoch_map: dict[int, object] = {}
        else:
            ds = self.dataseries.get(prefix)
            if select:
                self.dataseries.selected_name = prefix
            if ds.locked:
                # A locked series keeps its membership; copy it to change it
                nmh.history(
                    "dataseries '%s' is locked; not updated" % prefix,
                    path=self.path_str,
                    quiet=quiet,
                )
                return ds
            # Pre-populate so we only create genuinely new channels/epochs
            channel_map = {
                ch_name: ds.channels.get(ch_name) for ch_name in ds.channels
            }
            epoch_map = {}
            for ep_name in ds.epochs:
                try:
                    epoch_map[int(ep_name[1:])] = ds.epochs.get(ep_name)
                except (ValueError, IndexError):
                    pass

        channel_chars = sorted(set(ch for ch, _ in matches.keys()))
        epoch_nums = sorted(set(ep for _, ep in matches.keys()))

        # Create only channels/epochs not already present
        for ch_char in channel_chars:
            if ch_char not in channel_map:
                channel = ds.channels.new(quiet=True)
                if channel is not None:
                    channel_map[ch_char] = channel

        for ep_num in epoch_nums:
            if ep_num not in epoch_map:
                epoch = ds.epochs.new(quiet=True)
                if epoch is not None:
                    epoch_map[ep_num] = epoch

        # Link data to channels and epochs; skip if already linked
        for (ch_char, ep_num), data in matches.items():
            channel = channel_map.get(ch_char)
            epoch = epoch_map.get(ep_num)
            if channel is not None and data not in channel.data:
                channel.data.append(data)
            if epoch is not None and data not in epoch.data:
                epoch.data.append(data)

        return ds

    def new_dataseries(
        self,
        prefix: str,
        n_channels: int = 1,
        n_epochs: int = 1,
        n_points: int = 100,
        dx: float = 1.0,
        x_start: float = 0.0,
        x_label: str = "",
        x_units: str = "",
        y_label: str = "",
        y_units: str = "",
        fill=0.0,
        ch_start: int = 0,
        ep_start: int = 0,
        select: bool = False,
        quiet: bool = nmc.QUIET,
    ) -> NMDataSeries | None:
        """Create or extend an NMDataSeries with synthetic NumPy arrays.

        Generates ``n_channels * n_epochs`` NMData objects named
        ``{prefix}{chan}{epoch}`` (e.g. RecordA0, RecordA1, RecordB0 ...),
        fills each array according to ``fill``, then calls
        :meth:`sync_dataseries` which creates the dataseries if it does not
        yet exist, or extends it with any new channels and epochs if it does.

        Use ``ch_start`` and ``ep_start`` to append to an existing dataseries::

            folder.new_dataseries("Record", n_channels=2, n_epochs=5)
            # Later, append 3 more epochs to both channels:
            folder.new_dataseries("Record", n_channels=2, n_epochs=3, ep_start=5)

        Args:
            prefix: Data array name prefix (e.g. ``"Record"``). Must satisfy
                ``nmu.name_ok()``.
            n_channels: Number of channels to generate.
            n_epochs: Number of epochs to generate.
            n_points: Number of samples per array (0 creates empty arrays).
            dx: X-axis sample interval.
            x_start: X-axis start value (default ``0.0``).
            x_label: X-axis label (e.g. ``"Time"``).
            x_units: X-axis units (e.g. ``"ms"``).
            y_label: Y-axis label (e.g. ``"Voltage"``).
            y_units: Y-axis units (e.g. ``"mV"``).
            fill: How to populate each array.  Two forms are accepted:

                - **Scalar** (``int``, ``float``, or ``numpy.nan``) — passed
                  to ``numpy.full(n_points, fill)``.  Examples::

                      fill=0.0        # zeros (default)
                      fill=numpy.nan  # NaN
                      fill=1.0        # ones
                      fill=-9999.0    # sentinel value

                - **Callable** ``f(n_points) -> numpy.ndarray`` — called once
                  per array.  Any NumPy factory or custom function works::

                      fill=numpy.zeros
                      fill=numpy.ones
                      fill=numpy.random.random          # uniform [0, 1)
                      fill=lambda n: numpy.random.normal(0, 2.0, n)

            ch_start: Index of the first channel to generate (0 = A, 1 = B,
                ...). Default ``0``.
            ep_start: Index of the first epoch to generate. Default ``0``.
            select: Whether to select the dataseries after creation/update.
            quiet: If True, suppress history output.

        Returns:
            The NMDataSeries (new or updated), or None on failure.

        Raises:
            TypeError: If ``prefix`` is not a string, numeric args have wrong
                type, or ``fill`` is neither a scalar nor a callable.
            ValueError: If ``prefix`` fails name validation,
                ``n_channels`` / ``n_epochs`` > 0 violated,
                ``n_points`` / ``ch_start`` / ``ep_start`` < 0 violated, or
                any generated data name already exists.
        """
        import numpy as np

        if not isinstance(prefix, str):
            raise TypeError(nmu.type_error_str(prefix, "prefix", "string"))
        if not prefix or not nmu.name_ok(prefix):
            raise ValueError("prefix: %r" % prefix)
        for arg_name, arg_val, min_val in (
            ("n_channels", n_channels, 1),
            ("n_epochs",   n_epochs,   1),
            ("n_points",   n_points,   0),
            ("ch_start",   ch_start,   0),
            ("ep_start",   ep_start,   0),
        ):
            if isinstance(arg_val, bool) or not isinstance(arg_val, int):
                raise TypeError(nmu.type_error_str(arg_val, arg_name, "int"))
            if arg_val < min_val:
                raise ValueError(
                    "%s must be >= %d, got %d" % (arg_name, min_val, arg_val)
                )
        if not callable(fill) and not isinstance(fill, (int, float)):
            raise TypeError(
                "fill must be a numeric scalar or callable, got %r" % type(fill).__name__
            )

        ch_nums = range(ch_start, ch_start + n_channels)
        ep_nums = range(ep_start, ep_start + n_epochs)

        # Pre-flight: check data names don't conflict before touching any state
        for ch_i in ch_nums:
            ch_char = nmu.channel_char(ch_i)
            for ep in ep_nums:
                name = "%s%s%d" % (prefix, ch_char, ep)
                if name in self.data:
                    raise ValueError("data %r already exists" % name)

        xscale = {
            "start": x_start, "delta": dx, "label": x_label, "units": x_units,
        }
        yscale = {"label": y_label, "units": y_units}

        matches: dict[tuple[str, int], NMData] = {}
        for ch_i in ch_nums:
            ch_char = nmu.channel_char(ch_i)
            for ep in ep_nums:
                name = "%s%s%d" % (prefix, ch_char, ep)
                arr = fill(n_points) if callable(fill) else np.full(n_points, fill)
                data = self.data.new(
                    name=name, nparray=arr, xscale=xscale, yscale=yscale,
                    quiet=True,
                )
                if data is not None:
                    matches[(ch_char, ep)] = data

        is_new = prefix not in self.dataseries
        ds = self.build_dataseries(prefix, matches, select=select, quiet=True)
        if ds is None:
            return None

        channel_chars = sorted(set(ch for ch, _ in matches.keys()))
        epoch_nums = sorted(set(ep for _, ep in matches.keys()))
        ch_str = ", ".join(channel_chars)
        ep_str = ("E%d..E%d" % (epoch_nums[0], epoch_nums[-1])) if epoch_nums else ""
        action = "new" if is_new else "updated"
        nmh.history(
            "%s dataseries '%s': channels %s; epochs %s"
            % (action, prefix, ch_str, ep_str),
            path=self.path_str,
            quiet=quiet,
        )
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        nmch.add_nm_command(
            '%s.new_dataseries(%r, n_channels=%r, n_epochs=%r, n_points=%r, '
            'dx=%r, x_start=%r, x_label=%r, x_units=%r, y_label=%r, y_units=%r, '
            'fill=%r, ch_start=%r, ep_start=%r)'
            % (_p, prefix, n_channels, n_epochs, n_points,
               dx, x_start, x_label, x_units, y_label, y_units,
               fill, ch_start, ep_start)
        )
        return ds

    def _scan_dataseries(
        self,
        prefix: str,
    ) -> tuple[str, dict] | None:
        """Scan folder data for names matching *prefix* without creating anything.

        Returns:
            ``(actual_prefix, matches)`` where *matches* maps
            ``(channel_char, epoch_num)`` to :class:`NMData`, or ``None`` if
            no matching data is found.
        """
        matches: dict[tuple[str, int], NMData] = {}
        actual_prefix: str | None = None

        for name, data in sorted(self.data.items()):
            parsed = nmu.parse_data_name(name)
            if parsed is None:
                continue
            detected_prefix, channel_char, epoch_num = parsed
            if not nmu.match_prefix(prefix, detected_prefix):
                continue
            if actual_prefix is None:
                actual_prefix = detected_prefix
            elif actual_prefix != detected_prefix:
                continue
            matches[(channel_char, epoch_num)] = data

        if not matches or actual_prefix is None:
            return None
        return actual_prefix, matches

    def scan_dataseries(self, prefix: str) -> NMDataSeriesScan | None:
        """Return a preview of what sync_dataseries() would create, without side effects.

        Useful as a first step before committing to dataseries creation — e.g.
        a GUI can call this, show the user the detected channels and epochs, and
        only call :meth:`sync_dataseries` on confirmation.

        Args:
            prefix: Prefix or wildcard pattern (same as :meth:`sync_dataseries`).

        Returns:
            :class:`NMDataSeriesScan` with detected prefix, channels, epochs and
            data count, or ``None`` if no matching data is found.

        Example:
            >>> info = folder.scan_dataseries("Record")
            >>> info.prefix    # "Record"
            >>> info.channels  # ["A", "B"]
            >>> info.epochs    # [0, 1, 2]
            >>> info.n_data    # 6
        """
        if not prefix or not isinstance(prefix, str):
            return None
        result = self._scan_dataseries(prefix)
        if result is None:
            return None
        actual_prefix, matches = result
        channels = sorted(set(ch for ch, _ in matches.keys()))
        epochs = sorted(set(ep for _, ep in matches.keys()))
        return NMDataSeriesScan(
            prefix=actual_prefix,
            channels=channels,
            epochs=epochs,
            n_data=len(matches),
        )

    def sync_dataseries(
        self,
        prefix: str,
        select: bool = False,
        quiet: bool = nmc.QUIET,
    ) -> NMDataSeries | None:
        """Create or update a dataseries from data in the folder matching a prefix.

        Scans the data container for names matching the NeuroMatic pattern
        ``{prefix}{channel}{epoch}`` and creates or extends the dataseries:

        - If the dataseries does not yet exist, creates it with the matched
          channels and epochs.
        - If it already exists, adds any channels and epochs not already
          present; data already linked to existing channels/epochs is skipped.

        The prefix can be partial — ``"Rec"`` will match ``"RecordA0"``.
        The full detected prefix (e.g. ``"Record"``) becomes the dataseries name.

        Wildcard patterns (``*``, ``?``, ``[seq]``) are also supported via
        :mod:`fnmatch` — e.g. ``"Rec*"`` or ``"Avg_*"``.  When a wildcard
        matches more than one distinct prefix the alphabetically-first prefix
        is used.

        For a non-destructive preview use :meth:`scan_dataseries` first.

        Args:
            prefix: Prefix or wildcard pattern to match (case-insensitive).
            select: Whether to select the dataseries after creation/update.
            quiet: If True, suppress history output.

        Returns:
            The NMDataSeries (new or updated), or None if no matching data found.

        Example:
            >>> folder.sync_dataseries("Record")
            # Creates or updates dataseries "Record" from RecordA0, RecordB0 …
            >>> folder.sync_dataseries("Avg_*")
            # Matches "Avg_Record", "Avg_Test", etc. (alphabetically first wins)
        """
        if not prefix or not isinstance(prefix, str):
            return None

        result = self._scan_dataseries(prefix)
        if result is None:
            return None
        actual_prefix, matches = result

        is_new = actual_prefix not in self.dataseries
        ds = self.build_dataseries(actual_prefix, matches, select=select, quiet=True)
        if ds is None:
            return None

        channel_chars = sorted(set(ch for ch, _ in matches.keys()))
        epoch_nums = sorted(set(ep for _, ep in matches.keys()))
        ch_str = ", ".join(channel_chars)
        ep_str = ("E%d..E%d" % (epoch_nums[0], epoch_nums[-1])) if epoch_nums else ""
        action = "new" if is_new else "updated"
        nmh.history(
            "%s dataseries '%s': channels %s; epochs %s"
            % (action, actual_prefix, ch_str, ep_str),
            path=self.path_str,
            quiet=quiet,
        )
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        nmch.add_nm_command('%s.sync_dataseries(%r)' % (_p, actual_prefix))
        return ds

    def copy_dataseries(
        self,
        prefix: str,
        new_prefix: str | None = None,
        channel=None,
        epoch=None,
        folder: "NMFolder | None" = None,
        quiet: bool = nmc.QUIET,
    ) -> NMDataSeries | None:
        """Copy a dataseries to the same or a different folder.

        Deep-copies the NMData arrays for the specified channels and epochs,
        creates them in the target folder under *new_prefix*, then calls
        :meth:`sync_dataseries` to build the dataseries structure.

        Args:
            prefix: Source dataseries name.
            new_prefix: Destination prefix.  When copying within the same
                folder and *new_prefix* is ``None``, ``"C_"`` is prepended
                automatically.  When copying to a different folder and
                *new_prefix* is ``None``, *prefix* is reused.
            channel: Channels to copy — ``None`` (all), str, int, list, or
                range.
            epoch: Epochs to copy — ``None`` (all), str, int, list, or
                range.
            folder: Destination :class:`NMFolder`.  ``None`` means the same
                folder.
            quiet: Suppress history messages.

        Returns:
            The new :class:`NMDataSeries` in the target folder, or ``None``
            on failure.

        Raises:
            ValueError: If source dataseries not found, *new_prefix* already
                exists in the target folder, or any NMData name would conflict.
        """
        import numpy as np

        src_ds = self._get_dataseries(prefix)
        target = folder if folder is not None else self

        # Resolve new prefix
        if new_prefix is None:
            new_prefix = ("C_" + prefix) if (target is self) else prefix

        # Pre-flight: new_prefix must not already be a dataseries in target
        if target.dataseries.get(new_prefix) is not None:
            raise ValueError(
                "dataseries %r already exists in target folder" % new_prefix
            )

        # Resolve channel/epoch subsets
        if channel is None:
            ch_names = list(src_ds.channels)
        else:
            ch_names = self._resolve_channel_names(src_ds, channel)

        if epoch is None:
            ep_names = list(src_ds.epochs)
        else:
            ep_names = self._resolve_epoch_names(src_ds, epoch)

        # Map epoch names → epoch numbers for make_data_name
        ep_nums = []
        for ep_name in ep_names:
            try:
                ep_nums.append(int(ep_name[1:]))
            except (ValueError, IndexError):
                raise ValueError("cannot parse epoch number from %r" % ep_name)

        # Build a lookup: (ch_char, ep_num) → source NMData
        src_lookup: dict[tuple[str, int], object] = {}
        for ch_name in ch_names:
            ch = src_ds.channels.get(ch_name)
            for ep_name, ep_num in zip(ep_names, ep_nums):
                ep = src_ds.epochs.get(ep_name)
                # Find the NMData that belongs to both this channel and epoch
                src_data = None
                for d in ch.data:
                    if d in ep.data:
                        src_data = d
                        break
                if src_data is not None:
                    src_lookup[(ch_name, ep_num)] = src_data

        # Pre-flight: check for NMData name conflicts in target folder
        ch_indices = {ch: i for i, ch in enumerate(ch_names)}
        new_names = []
        for ch_name in ch_names:
            ch_idx = ch_indices[ch_name]
            for ep_num in ep_nums:
                from pyneuromatic.io.base import make_data_name
                name = make_data_name(new_prefix, ch_idx, ep_num)
                if target.data.get(name) is not None:
                    raise ValueError(
                        "NMData name %r already exists in target folder" % name
                    )
                new_names.append(name)

        # Copy data into target folder and build matches dict directly
        from pyneuromatic.io.base import make_data_name
        new_matches: dict[tuple[str, int], NMData] = {}
        brought: dict = {}  # x arrays copied into the target folder, once each

        for ch_name in ch_names:
            ch_idx = ch_indices[ch_name]
            new_ch_char = nmu.channel_char(ch_idx)
            for ep_num in ep_nums:
                name = make_data_name(new_prefix, ch_idx, ep_num)
                src_data = src_lookup.get((ch_name, ep_num))
                arr = (
                    src_data.nparray.copy()
                    if src_data is not None and src_data.nparray is not None
                    else np.array([])
                )
                # Read xscale/yscale from the source NMData object; the
                # channel xscale is a separate object not propagated by
                # build_dataseries
                xscale_dict = src_data.xscale.to_dict() if src_data is not None else {}
                yscale_dict = src_data.yscale.to_dict() if src_data is not None else {}
                new_data = target.data.new(
                    name,
                    nparray=arr,
                    xscale=xscale_dict if xscale_dict else None,
                    yscale=yscale_dict if yscale_dict else None,
                    quiet=True,
                )
                if new_data is not None:
                    new_matches[(new_ch_char, ep_num)] = new_data
                    # x values: the same x array, in the target folder
                    if src_data is not None and src_data.xarray_source is not None:
                        new_data.set_xarray_from(
                            target.data.bring_xarray(src_data.xarray_source, brought))
                    elif src_data is not None and src_data.xarray is not None:
                        new_data.xarray = src_data.xarray.copy()

        ds = target.build_dataseries(new_prefix, new_matches, quiet=True)
        if ds is not None:
            self._copy_series_settings(src_ds, ds, ch_names, ep_names)

        n_ch = len(ch_names)
        n_ep = len(ep_nums)
        src_desc = "'%s' -> '%s'" % (prefix, new_prefix)
        if target is not self:
            src_desc += " (to folder '%s')" % target.name
        nmh.history(
            "copied dataseries %s: %d channel(s), %d epoch(s)"
            % (src_desc, n_ch, n_ep),
            path=self.path_str,
            quiet=quiet,
        )
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        folder_str = ('%s["%s"]' % (folder._nm_cmd_path, folder.name)) if folder is not None else "None"
        nmch.add_nm_command(
            '%s.copy_dataseries(%r, new_prefix=%r, channel=%r, epoch=%r, folder=%s)'
            % (_p, prefix, new_prefix, channel, epoch, folder_str)
        )
        return ds

    def remove_dataseries(
        self,
        prefix: str,
        delete_data: bool = False,
        quiet: bool = nmc.QUIET,
    ) -> None:
        """Remove a dataseries from this folder.

        Args:
            prefix: Name of the dataseries to remove.
            delete_data: If True, also delete all NMData objects that belong
                to this dataseries from the folder's data container.
            quiet: Suppress history messages.

        Raises:
            ValueError: If no dataseries with *prefix* exists.
        """
        ds = self._get_unlocked_dataseries(prefix)

        if delete_data:
            # Collect unique data names across all channels
            data_names: set[str] = set()
            for ch_name in ds.channels:
                ch = ds.channels.get(ch_name)
                for d in list(ch.data):
                    data_names.add(d.name)
            for name in data_names:
                self.data.pop(name, default=None, quiet=True)

        self.dataseries.pop(prefix, quiet=True)
        msg = "removed dataseries '%s'" % prefix
        if delete_data:
            msg += " (and %d data objects)" % len(data_names)
        nmh.history(msg, path=self.path_str, quiet=quiet)
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        nmch.add_nm_command(
            '%s.remove_dataseries(%r, delete_data=%r)' % (_p, prefix, delete_data)
        )

    def remove_dataseries_channel(
        self,
        prefix: str,
        channel,
        delete_data: bool = False,
        quiet: bool = nmc.QUIET,
    ) -> None:
        """Remove one or more channels from a dataseries.

        Args:
            prefix: Name of the dataseries.
            channel: Channel(s) to remove — str ('A'), int (0), list, or range.
            delete_data: If True, delete the associated NMData from the folder.
            quiet: Suppress history messages.

        Raises:
            ValueError: If dataseries or channel not found.
            TypeError: If *channel* has an unsupported type.
        """
        ds = self._get_unlocked_dataseries(prefix)
        ch_names = self._resolve_channel_names(ds, channel)
        n_deleted = 0

        for ch_name in ch_names:
            ch = ds.channels.get(ch_name)
            data_list = list(ch.data)

            if delete_data:
                for d in data_list:
                    self.data.pop(d.name, default=None, quiet=True)
                n_deleted += len(data_list)

            # Remove data refs from all epochs
            for ep_name in ds.epochs:
                ep = ds.epochs.get(ep_name)
                for d in data_list:
                    if d in ep.data:
                        ep.data.remove(d)

            ds.channels.pop(ch_name, quiet=True)

        ch_str = ", ".join(ch_names)
        msg = "removed channel(s) [%s] from dataseries '%s'" % (ch_str, prefix)
        if delete_data:
            msg += " (and %d data objects)" % n_deleted
        nmh.history(msg, path=self.path_str, quiet=quiet)
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        nmch.add_nm_command(
            '%s.remove_dataseries_channel(%r, %r, delete_data=%r)'
            % (_p, prefix, channel, delete_data)
        )

    def remove_dataseries_epoch(
        self,
        prefix: str,
        epoch,
        delete_data: bool = False,
        quiet: bool = nmc.QUIET,
    ) -> None:
        """Remove one or more epochs from a dataseries.

        Args:
            prefix: Name of the dataseries.
            epoch: Epoch(s) to remove — str ('E0'), int (0), list, or range.
            delete_data: If True, delete the associated NMData from the folder.
            quiet: Suppress history messages.

        Raises:
            ValueError: If dataseries or epoch not found.
            TypeError: If *epoch* has an unsupported type.
        """
        ds = self._get_unlocked_dataseries(prefix)
        ep_names = self._resolve_epoch_names(ds, epoch)
        n_deleted = 0

        for ep_name in ep_names:
            ep = ds.epochs.get(ep_name)
            data_list = list(ep.data)

            if delete_data:
                for d in data_list:
                    self.data.pop(d.name, default=None, quiet=True)
                n_deleted += len(data_list)

            # Remove data refs from all channels
            for ch_name in ds.channels:
                ch = ds.channels.get(ch_name)
                for d in data_list:
                    if d in ch.data:
                        ch.data.remove(d)

            ds.epochs.pop(ep_name, quiet=True)

        if len(ep_names) == 1:
            ep_str = ep_names[0]
        else:
            ep_str = "%s..%s (%d epochs)" % (ep_names[0], ep_names[-1], len(ep_names))
        msg = "removed epoch(s) [%s] from dataseries '%s'" % (ep_str, prefix)
        if delete_data:
            msg += " (and %d data objects)" % n_deleted
        nmh.history(msg, path=self.path_str, quiet=quiet)
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        nmch.add_nm_command(
            '%s.remove_dataseries_epoch(%r, %r, delete_data=%r)'
            % (_p, prefix, epoch, delete_data)
        )

    @staticmethod
    def _fill_array(fill, n_points: int):
        """An array of *n_points* from *fill* (a number, or f(n) -> array)."""
        import numpy as np

        if callable(fill):
            return np.asarray(fill(n_points), dtype=float)
        if isinstance(fill, bool) or not isinstance(fill, (int, float)):
            raise TypeError(
                "fill must be a numeric scalar or callable, got %r" % type(fill).__name__
            )
        return np.full(n_points, float(fill))

    @staticmethod
    def _check_count(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, name, "int"))
        if value < 1:
            raise ValueError("%s must be >= 1, got %d" % (name, value))

    @staticmethod
    def _epoch_number(ds: NMDataSeries, epoch) -> int | None:
        """The number of an epoch named e.g. 'E5', or None."""
        text = epoch.name[len(ds.epochs.auto_name_prefix):]
        return int(text) if text.isdigit() else None

    def add_dataseries_channels(
        self,
        prefix: str,
        n_channels: int = 1,
        fill=0.0,
        y_label: str = "",
        y_units: str = "",
        quiet: bool = nmc.QUIET,
    ) -> list:
        """Add channels to a dataseries: one new array per epoch.

        New channels take the letters after the last channel (e.g. C, D
        after A, B); their arrays are named ``{prefix}{channel}{epoch}``
        (e.g. RecordC0) and take the points and x scale of the epoch's
        existing arrays.

        Args:
            prefix: Name of the dataseries.
            n_channels: How many channels to add.
            fill: Initial values: a number (e.g. 0 or NaN) or a function
                ``f(n_points) -> array`` (e.g. noise).
            y_label: Y label of the new arrays (e.g. ``"Im"``).
            y_units: Y units of the new arrays (e.g. ``"pA"``).
            quiet: Suppress history messages.

        Returns:
            The new NMChannels.

        Raises:
            RuntimeError: If the dataseries is locked.
            ValueError: If the dataseries is missing, has no epochs, there
                are no letters left, or an array name exists.
        """
        from pyneuromatic.core.nm_channel import NMChannel

        ds = self._get_unlocked_dataseries(prefix)
        self._check_count("n_channels", n_channels)
        epochs = list(ds.epochs.values())
        if not epochs:
            raise ValueError("dataseries '%s' has no epochs" % prefix)
        used = [nmu.CHANNEL_CHARS.index(c) for c in ds.channels.keys()
                if c in nmu.CHANNEL_CHARS]
        first = max(used) + 1 if used else 0
        if first + n_channels > len(nmu.CHANNEL_CHARS):
            raise ValueError(
                "dataseries '%s' can have at most %d channels (A to Z)"
                % (prefix, len(nmu.CHANNEL_CHARS))
            )
        letters = [nmu.channel_char(i) for i in range(first, first + n_channels)]
        # Each epoch's new arrays copy the size and x scale of one it has
        templates = {}
        for epoch in epochs:
            if not epoch.data:
                raise ValueError("epoch '%s' has no arrays to copy the size of" % epoch.name)
            templates[epoch.name] = epoch.data[0]
        number = {e.name: e.name[len(ds.epochs.auto_name_prefix):] for e in epochs}
        names = ["%s%s%s" % (prefix, c, number[e.name]) for c in letters for e in epochs]
        for name in names:  # before changing anything
            if name in self.data:
                raise ValueError("data %r already exists" % name)

        yscale = {"label": y_label, "units": y_units}
        added = []
        for letter in letters:
            first_template = templates[epochs[0].name]
            channel = NMChannel(parent=ds, name=letter,
                                xscale=first_template.xscale.to_dict(), yscale=yscale)
            ds.channels._add(channel, quiet=True)
            for epoch in epochs:
                template = templates[epoch.name]
                n_points = template.nparray.size if template.nparray is not None else 0
                data = self.data.new(
                    name="%s%s%s" % (prefix, letter, number[epoch.name]),
                    nparray=self._fill_array(fill, n_points),
                    xscale=template.xscale.to_dict(), yscale=yscale, quiet=True,
                )
                channel.data.append(data)
                epoch.data.append(data)
            added.append(channel)

        nmh.history(
            "added channel(s) %s to dataseries '%s' (%d arrays)"
            % (", ".join(letters), prefix, len(names)),
            path=self.path_str, quiet=quiet,
        )
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        nmch.add_nm_command(
            '%s.add_dataseries_channels(%r, n_channels=%r, fill=%r, y_label=%r, y_units=%r)'
            % (_p, prefix, n_channels, fill, y_label, y_units)
        )
        return added

    def add_dataseries_epochs(
        self,
        prefix: str,
        n_epochs: int = 1,
        fill=0.0,
        quiet: bool = nmc.QUIET,
    ) -> list:
        """Add epochs to a dataseries: one new array per channel.

        New epochs are numbered after the last epoch (e.g. E10, E11 after
        E9); their arrays are named ``{prefix}{channel}{epoch}`` (e.g.
        RecordA10) and take the points and scales of the channel's last
        array.

        Args:
            prefix: Name of the dataseries.
            n_epochs: How many epochs to add.
            fill: Initial values: a number (e.g. 0 or NaN) or a function
                ``f(n_points) -> array`` (e.g. noise).
            quiet: Suppress history messages.

        Returns:
            The new NMEpochs.

        Raises:
            RuntimeError: If the dataseries is locked.
            ValueError: If the dataseries is missing, has no channels (or a
                channel has no arrays), or an array name exists.
        """
        from pyneuromatic.core.nm_epoch import NMEpoch

        ds = self._get_unlocked_dataseries(prefix)
        self._check_count("n_epochs", n_epochs)
        channels = list(ds.channels.values())
        if not channels:
            raise ValueError("dataseries '%s' has no channels" % prefix)
        for channel in channels:
            if not channel.data:
                raise ValueError("channel '%s' has no arrays to copy the size of"
                                 % channel.name)
        numbers = [n for n in (self._epoch_number(ds, e) for e in ds.epochs.values())
                   if n is not None]
        first = max(numbers) + 1 if numbers else 0
        new_numbers = list(range(first, first + n_epochs))
        names = ["%s%s%d" % (prefix, c.name, n) for n in new_numbers for c in channels]
        for name in names:  # before changing anything
            if name in self.data:
                raise ValueError("data %r already exists" % name)

        epoch_prefix = ds.epochs.auto_name_prefix
        added = []
        for number in new_numbers:
            epoch = NMEpoch(parent=ds, name="%s%d" % (epoch_prefix, number), number=number)
            ds.epochs._add(epoch, quiet=True)
            for channel in channels:
                template = channel.data[-1]
                n_points = template.nparray.size if template.nparray is not None else 0
                data = self.data.new(
                    name="%s%s%d" % (prefix, channel.name, number),
                    nparray=self._fill_array(fill, n_points),
                    xscale=template.xscale.to_dict(), yscale=template.yscale.to_dict(),
                    quiet=True,
                )
                channel.data.append(data)
                epoch.data.append(data)
            added.append(epoch)

        ep_str = ("%s%d" % (epoch_prefix, first) if n_epochs == 1 else
                  "%s%d..%s%d" % (epoch_prefix, first, epoch_prefix, new_numbers[-1]))
        nmh.history(
            "added epoch(s) %s to dataseries '%s' (%d arrays)" % (ep_str, prefix, len(names)),
            path=self.path_str, quiet=quiet,
        )
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        nmch.add_nm_command(
            '%s.add_dataseries_epochs(%r, n_epochs=%r, fill=%r)'
            % (_p, prefix, n_epochs, fill)
        )
        return added

    # --- moving and appending data series -------------------------------------

    @staticmethod
    def _series_arrays(ds: NMDataSeries) -> list:
        """(epoch, channel, data) for each array of *ds*, epoch by epoch."""
        found = []
        for epoch in ds.epochs.values():
            for channel in ds.channels.values():
                data = next((d for d in channel.data if any(d is e for e in epoch.data)), None)
                if data is not None:
                    found.append((epoch, channel, data))
        return found

    def _check_free_to_move(self, arrays: list) -> None:
        """Refuse arrays that others take their x values from (they'd leave
        those behind)."""
        moving = {id(d) for d in arrays}
        for data in arrays:
            users = [u for u in self.data.xarray_users(data.name) if id(u) not in moving]
            if users:
                raise RuntimeError(
                    "'%s' is the x array of '%s', which stays in folder '%s'"
                    % (data.name, users[0].name, self.name)
                )

    def _move_array(self, data: NMData, target: NMFolder, name: str) -> NMData:
        """Take *data* out of this folder into *target*, named *name*."""
        self.data.pop(data.name, quiet=True)
        data._name_set(newname=name, quiet=True)
        data._parent = target
        target.data._add(data, quiet=True)
        return data

    @staticmethod
    def _copy_sets(source_sets, target_sets, rename: dict[str, str]) -> None:
        """Copy *source_sets* into *target_sets*: each set's members, by their
        new names (*rename*); equation sets once their two sets exist."""
        equations = []
        for key in source_sets.keys():
            value = source_sets._map[key]
            if source_sets.tuple_is_equation(value):
                equations.append((key, value))
                continue
            members = [rename[m] for m in value if m in rename]
            target_sets.add(key, members, quiet=True)
        for key, value in equations:
            if target_sets._getkey(key) is None and all(
                target_sets._getkey(name) is not None for name in value[1:]
            ):
                target_sets._setitem(key, value)

    def _move_xarray_plan(self, ds: NMDataSeries, arrays: list) -> list:
        """The x arrays of the arrays of *ds* (not themselves in *ds*), each
        as (x array, "move" or "copy", names of other arrays using it): it
        moves with the series unless arrays staying here use it too."""
        moving = {id(d) for d in arrays}
        sources: dict = {}
        for data in arrays:
            source = data.xarray_source
            if source is not None and id(source) not in moving:
                sources[id(source)] = source
        plan = []
        for source in sources.values():
            others = []
            if source._parent is self and source.name in self.data:
                others = [u.name for u in self.data.xarray_users(source.name)
                          if id(u) not in moving]
            here = source._parent is self and source.name in self.data
            plan.append((source, "move" if here and not others else "copy", others))
        return plan

    def move_dataseries_note(self, prefix: str) -> str:
        """What happens to the series' x array(s) when it moves, e.g.
        "X array 'Times' moves too", or "" if it has none."""
        ds = self.dataseries.get(prefix)
        if ds is None:
            return ""
        notes = []
        for source, action, others in self._move_xarray_plan(ds, [
                d for _e, _c, d in self._series_arrays(ds)]):
            if action == "move":
                notes.append("X array '%s' moves too." % source.name)
            else:
                notes.append("A copy of x array '%s' goes too (it stays here for %s)."
                             % (source.name, ", ".join("'%s'" % n for n in others[:3])))
        return " ".join(notes)

    def _copy_series_settings(self, src: NMDataSeries, ds: NMDataSeries,
                              ch_names: list[str], ep_names: list[str]) -> None:
        """Give copy *ds* the epoch sets, groups, channel transforms and notes
        of *src*, for the channels and epochs copied (matched in order: the
        copy numbers them afresh)."""
        by_number = sorted(ep_names, key=lambda name: int(name[1:]))
        epoch_name = dict(zip(by_number, ds.epochs.keys()))
        self._copy_sets(src.epochs.sets, ds.epochs.sets, epoch_name)
        for old, new in epoch_name.items():
            group = src.epochs.groups.get_group(old)
            if group is not None:
                ds.epochs.groups.assign(new, group, quiet=True)
        for index, old in enumerate(ch_names):
            channel = ds.channels.get(nmu.channel_char(index))
            source = src.channels.get(old)
            if channel is not None and source is not None and source.transform:
                channel.transform = [t.to_dict() for t in source.transform]
        for entry in src.notes:
            ds.notes._entries.append(dict(entry))

    def move_data_plan(self, name: str, target: NMFolder) -> tuple:
        """Check moving data *name* to *target* (see :meth:`move_data`)
        without changing anything; returns (data, x array or None, "move",
        "copy" or "reuse").

        Raises the errors :meth:`move_data` describes (except a name clash)."""
        if not isinstance(target, NMFolder):
            raise TypeError(nmu.type_error_str(target, "target", "NMFolder"))
        if target is self:
            raise ValueError("'%s' is already in folder '%s'" % (name, self.name))
        key = self.data._getkey(name)
        if key is None:
            raise KeyError("data '%s' not found" % name)
        data = self.data[key]
        series = self.data.dataseries_of(key)
        if series:
            raise RuntimeError(
                "'%s' is part of data series '%s'; move the whole series with "
                "Prefix ⋯ › Move to Folder" % (key, series[0].name))
        in_use = self.data.xarray_in_use_text(key)
        if in_use:
            raise RuntimeError(in_use.replace("cannot be deleted", "cannot be moved"))
        source = data.xarray_source
        if source is None:
            return data, None, None
        here = source._parent is self and source.name in self.data
        others = [u for u in self.data.xarray_users(source.name) if u is not data] \
            if here else []
        there = target.data.get(source.name) if source.name in target.data else None
        if there is not None:
            from pyneuromatic.core.nm_data import _eq_arrays
            if not _eq_arrays(there.nparray, source.nparray):
                raise ValueError(
                    "folder '%s' already has a different '%s' (the x array of '%s')"
                    % (target.name, source.name, key))
            return data, source, "reuse"
        return data, source, "move" if here and not others else "copy"

    def move_data_note(self, name: str, target: NMFolder) -> str:
        """What happens to the array's x array when it moves, or ""."""
        try:
            _data, source, action = self.move_data_plan(name, target)
        except (KeyError, ValueError, RuntimeError, TypeError):
            return ""
        if source is None:
            return ""
        return {"move": "X array '%s' moves too.",
                "copy": "A copy of x array '%s' goes too (others here still use it).",
                "reuse": "It uses the '%s' already in that folder."}[action] % source.name

    def move_data(
        self,
        name: str,
        target: NMFolder,
        new_name: str | None = None,
        quiet: bool = nmc.QUIET,
    ) -> NMData:
        """Move data *name* (the same array, not a copy) to folder *target*,
        named *new_name* or as before. Its x array goes too (moved if
        nothing else here uses it, else copied; an identical one already in
        *target* is used instead).

        Raises:
            RuntimeError: If the array is part of a data series (move the
                series), or other arrays take their x values from it.
            ValueError: If *target* is this folder, the name is taken there,
                or *target* has a different array named as its x array.
        """
        data, source, action = self.move_data_plan(name, target)
        new_name = new_name or data.name
        if new_name in target.data:
            raise ValueError("folder '%s' already has data '%s'" % (target.name, new_name))
        if not nmu.name_ok(new_name):
            raise ValueError("name: %r" % new_name)
        if source is not None:
            data._NMData__xarray_source = None  # let go while it moves
            if action == "move":
                self._move_array(source, target, source.name)
                new_source = source
            elif action == "reuse":
                new_source = target.data[source.name]
            else:
                new_source = target.data.bring_xarray(source)
        old_name = data.name
        self._move_array(data, target, new_name)
        if source is not None:
            data._NMData__xarray_source = new_source
        nmh.history(
            "moved data '%s' to folder '%s'%s" % (
                old_name, target.name, " as '%s'" % new_name if new_name != old_name else ""),
            path=self.path_str, quiet=quiet,
        )
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        nmch.add_nm_command('%s.move_data(%r, nm.folders[%r], new_name=%r)'
                            % (_p, old_name, target.name, new_name))
        return data

    def move_dataseries(
        self,
        prefix: str,
        target: NMFolder,
        new_prefix: str | None = None,
        quiet: bool = nmc.QUIET,
    ) -> NMDataSeries:
        """Move data series *prefix*, with its arrays, epoch sets, groups,
        channel scales and transforms and notes, to folder *target*.

        With *new_prefix*, the series and its arrays are renamed on the way
        (e.g. RecordA0 -> Record2A0), e.g. when names clash in *target*.
        Arrays stay the same objects (no copy). An x array outside the
        series stays where it is, still used by the moved arrays.

        Raises:
            RuntimeError: If the series is locked, or one of its arrays is the
                x array of an array that stays behind.
            ValueError: If *target* is this folder, or a name already exists
                in *target*.
        """
        from pyneuromatic.core.nm_channel import NMChannel
        from pyneuromatic.core.nm_epoch import NMEpoch

        ds = self._get_unlocked_dataseries(prefix)
        if not isinstance(target, NMFolder):
            raise TypeError(nmu.type_error_str(target, "target", "NMFolder"))
        if target is self:
            raise ValueError("data series '%s' is already in folder '%s'" % (prefix, self.name))
        name = new_prefix or prefix
        if not nmu.name_ok(name):
            raise ValueError("prefix: %r" % name)
        if name in target.dataseries:
            raise ValueError("folder '%s' already has a data series '%s'" % (target.name, name))
        arrays = self._series_arrays(ds)
        renamed = {d.name: name + d.name[len(prefix):] if d.name.startswith(prefix) else d.name
                   for _e, _c, d in arrays}
        clash = next((n for n in renamed.values() if n in target.data), None)
        if clash is not None:
            raise ValueError("folder '%s' already has data '%s'" % (target.name, clash))
        self._check_free_to_move([d for _e, _c, d in arrays])
        # x arrays go with the series (moved, or copied if used here too):
        # a folder keeps its own x arrays
        xplan = self._move_xarray_plan(ds, [d for _e, _c, d in arrays])
        for source, _action, _others in xplan:
            if source.name in target.data or source.name in renamed.values():
                raise ValueError(
                    "folder '%s' already has data '%s' (the series' x array)"
                    % (target.name, source.name))

        # The series in its new folder: channels, epochs, sets, groups, notes
        moved = target.dataseries.new(name=name, quiet=True)
        for channel in ds.channels.values():
            moved.channels._add(NMChannel(
                parent=moved, name=channel.name, xscale=channel.xscale.to_dict(),
                yscale=channel.yscale.to_dict(),
                transform=[t.to_dict() for t in channel.transform]), quiet=True)
        for epoch in ds.epochs.values():
            moved.epochs._add(NMEpoch(parent=moved, name=epoch.name, number=epoch.number),
                              quiet=True)
        epoch_names = {e: e for e in ds.epochs.keys()}
        self._copy_sets(ds.epochs.sets, moved.epochs.sets, epoch_names)
        for epoch_name in ds.epochs.keys():
            group = ds.epochs.groups.get_group(epoch_name)
            if group is not None:
                moved.epochs.groups.assign(epoch_name, group, quiet=True)
        for entry in ds.notes:
            moved.notes._entries.append(dict(entry))

        # The arrays themselves, then the old series. The series' arrays let
        # go of their x arrays while those move (or are copied), then use
        # the ones in the new folder
        self.dataseries.pop(prefix, quiet=True)
        old_xarray = {id(d): d.xarray_source for _e, _c, d in arrays}
        for _e, _c, data in arrays:
            if old_xarray[id(data)] is not None:
                data._NMData__xarray_source = None
        new_xarray = {}
        for source, action, _others in xplan:
            if action == "move":
                self._move_array(source, target, source.name)
                new_xarray[id(source)] = source
            else:
                new_xarray[id(source)] = target.data.bring_xarray(source)
        for epoch, channel, data in arrays:
            self._move_array(data, target, renamed[data.name])
            source = old_xarray[id(data)]
            if source is not None:
                data._NMData__xarray_source = new_xarray.get(id(source), source)
            moved.channels[channel.name].data.append(data)
            moved.epochs[epoch.name].data.append(data)

        nmh.history(
            "moved dataseries '%s' (%d arrays) to folder '%s'%s"
            % (prefix, len(arrays), target.name,
               " as '%s'" % name if name != prefix else ""),
            path=self.path_str, quiet=quiet,
        )
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        nmch.add_nm_command(
            '%s.move_dataseries(%r, nm.folders[%r], new_prefix=%r)'
            % (_p, prefix, target.name, new_prefix)
        )
        return moved

    def _append_plan(self, prefix: str, source_prefix: str, source_folder: NMFolder,
                     delete_source: bool) -> tuple:
        """Check an append (see :meth:`append_dataseries`) without changing
        anything; returns (series, source, first epoch number, new epoch
        names by source epoch, source arrays, new array names by id).

        Raises the errors :meth:`append_dataseries` describes."""
        ds = self._get_unlocked_dataseries(prefix)
        if delete_source:
            src = source_folder._get_unlocked_dataseries(source_prefix)
        else:
            src = source_folder.dataseries.get(source_prefix)
            if src is None:
                raise ValueError("data series '%s' not found" % source_prefix)
        if src is ds:
            raise ValueError("can't append data series '%s' to itself" % prefix)
        channels, src_channels = list(ds.channels.keys()), list(src.channels.keys())
        if sorted(channels) != sorted(src_channels):
            raise ValueError(
                "the channels differ: '%s' has %s, '%s' has %s"
                % (prefix, ", ".join(channels) or "none",
                   source_prefix, ", ".join(src_channels) or "none")
            )
        numbers = [n for n in (self._epoch_number(ds, e) for e in ds.epochs.values())
                   if n is not None]
        first = max(numbers) + 1 if numbers else 0
        epoch_prefix = ds.epochs.auto_name_prefix
        new_epoch = {e: "%s%d" % (epoch_prefix, first + i)
                     for i, e in enumerate(src.epochs.keys())}
        arrays = self._series_arrays(src)
        new_name = {id(d): "%s%s%s" % (prefix, c.name, new_epoch[e.name][len(epoch_prefix):])
                    for e, c, d in arrays}
        clash = next((n for n in new_name.values() if n in self.data), None)
        if clash is not None:
            raise ValueError("data '%s' already exists" % clash)
        if delete_source:
            source_folder._check_free_to_move([d for _e, _c, d in arrays])
        # The appended arrays take this series' x array (one setting per
        # series, in this folder); they must fit it
        xarray = self._series_xarray(ds)
        if xarray is not None:
            for _e, _c, data in arrays:
                if data.nparray is not None and data.nparray.size != xarray.nparray.size:
                    raise ValueError(
                        "'%s' has %d points, but the x array of '%s', '%s', has %d"
                        % (data.name, data.nparray.size, prefix, xarray.name,
                           xarray.nparray.size))
        return ds, src, first, new_epoch, arrays, new_name

    @staticmethod
    def _series_xarray(ds: NMDataSeries):
        """The x array all the arrays of *ds* share, or None (none, or mixed)."""
        sources = {id(d.xarray_source): d.xarray_source
                   for c in ds.channels.values() for d in c.data}
        return next(iter(sources.values())) if len(sources) == 1 else None

    def append_dataseries_note(self, prefix: str, source_prefix: str,
                               source_folder: NMFolder | None = None) -> str:
        """How the appended arrays get their x values, when the two series
        differ (e.g. "The appended arrays use Start and Delta, as 'Record'
        does ('Stim' used x array 'T2')"), else ""."""
        source_folder = source_folder if source_folder is not None else self
        ds, src = self.dataseries.get(prefix), source_folder.dataseries.get(source_prefix)
        if ds is None or src is None:
            return ""
        mine, theirs = self._series_xarray(ds), self._series_xarray(src)
        mine_name = mine.name if mine is not None else None
        theirs_name = theirs.name if theirs is not None else None
        if mine_name == theirs_name and (mine is None or source_folder is self):
            return ""
        how = "x array '%s'" % mine_name if mine_name else "Start and Delta"
        was = "x array '%s'" % theirs_name if theirs_name else "Start and Delta"
        return ("The appended arrays use %s, as '%s' does ('%s' used %s)."
                % (how, prefix, source_prefix, was))

    def append_dataseries(
        self,
        prefix: str,
        source_prefix: str,
        source_folder: NMFolder | None = None,
        delete_source: bool = False,
        quiet: bool = nmc.QUIET,
    ) -> list:
        """Add the epochs of data series *source_prefix* (in *source_folder*,
        default this folder) to the end of data series *prefix*.

        The appended epochs are numbered after the last epoch of *prefix*
        and their arrays renamed to match (e.g. StimA0 -> RecordA10); their
        epoch sets and groups come along. With *delete_source*, the source
        series' arrays are moved and the series removed; otherwise they are
        copied and the source is kept.

        Returns:
            The new NMEpochs.

        Raises:
            RuntimeError: If *prefix* is locked, or *delete_source* and the
                source is locked.
            ValueError: If the channels differ, the source is the same series,
                or a new array name already exists.
        """
        from pyneuromatic.core.nm_epoch import NMEpoch

        source_folder = source_folder if source_folder is not None else self
        ds, src, first, new_epoch, arrays, new_name = self._append_plan(
            prefix, source_prefix, source_folder, delete_source)
        xarray = self._series_xarray(ds)

        added = []
        for src_epoch in src.epochs.values():
            name = new_epoch[src_epoch.name]
            epoch = NMEpoch(parent=ds, name=name, number=first + len(added))
            ds.epochs._add(epoch, quiet=True)
            added.append(epoch)
        if delete_source:
            source_folder.dataseries.pop(source_prefix, quiet=True)
        for src_epoch, channel, data in arrays:
            name = new_name[id(data)]
            if delete_source:
                data = source_folder._move_array(data, self, name)
            else:
                data = self._add_array_copy(data, name)
            # x values: this series' x array (in this folder), or none
            data._NMData__xarray_source = None
            data.set_xarray_from(xarray)
            ds.channels[channel.name].data.append(data)
            ds.epochs[new_epoch[src_epoch.name]].data.append(data)
        # Epoch sets and groups of the appended epochs
        self._copy_sets(src.epochs.sets, ds.epochs.sets, new_epoch)
        for src_name, name in new_epoch.items():
            group = src.epochs.groups.get_group(src_name)
            if group is not None:
                ds.epochs.groups.assign(name, group, quiet=True)

        last = new_epoch[list(new_epoch)[-1]] if new_epoch else ""
        nmh.history(
            "appended dataseries '%s' (%d epochs, %s to %s) to '%s'%s"
            % (source_prefix, len(added), new_epoch[list(new_epoch)[0]] if new_epoch else "",
               last, prefix, " (source removed)" if delete_source else ""),
            path=self.path_str, quiet=quiet,
        )
        _p = '%s["%s"]' % (self._nm_cmd_path, self.name)
        nmch.add_nm_command(
            '%s.append_dataseries(%r, %r, delete_source=%r)'
            % (_p, prefix, source_prefix, delete_source)
        )
        return added

    def _add_array_copy(self, data: NMData, name: str) -> NMData:
        """A copy of *data* in this folder, named *name* (it keeps any x
        array, as Duplicate does)."""
        result = copy.deepcopy(data)
        result._name_set(newname=name, quiet=True)
        result._parent = self
        self.data._add(result, quiet=True)
        return result


class NMFolderContainer(NMObjectContainer):
    """
    Container of NMFolders
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMFolderContainer0",
        rename_on: bool = True,
        name_prefix: str = "folder",
        name_seq_format: str = "0",
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            rename_on=rename_on,
            auto_name_prefix=name_prefix,
            auto_name_seq_format=name_seq_format,
        )

    # override
    @property
    def _nm_cmd_path(self) -> str:
        return "folders"

    # override, no super
    def content_type(self) -> str:
        return NMFolder.__name__

    # override
    def new(
        self,
        name: str | None = None,
        select: bool = False,
        quiet: bool = nmc.QUIET
    ) -> NMFolder | None:
        actual_name = self._newkey(name)
        # Use self._parent (NMManager) to skip container in parent chain,
        # consistent with NMData, NMDataSeries, NMChannel, NMEpoch
        f = NMFolder(parent=self._parent, name=actual_name)
        if super()._add(f, select=select, quiet=quiet):
            return f
        return None

    def open_hdf5(self):
        dataseries = "Record"
        with h5py.File("nmFolder0.hdf5", "r") as f:
            # print(f.keys())
            data = []
            for k in f.keys():
                if k[0 : len(dataseries)] == dataseries:
                    print(k)
            # for name in f:
            # print(name)
            d = f["RecordA0"]

            for i in d.attrs.keys():
                print(i)
            # cannot get access to attribute values for keys:
            # probably need to update h5py to v 2.10
            # IGORWaveNote
            # IGORWaveType
            # print(d.attrs.__getitem__('IGORWaveNote'))
            # for a in d.attrs:
            # print(item + ":", d.attrs[item])
            # print(item + ":", d.attrs.get(item))
            # print(a.shape)
            # for k in a.keys():
            # print(k)
            # print(a)
            # pf = f['NMPrefix_Record']
            # print(pf)
