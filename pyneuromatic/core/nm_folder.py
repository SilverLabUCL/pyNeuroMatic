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
import datetime
from typing import Any
import h5py

from pyneuromatic.core.nm_data import NMDataContainer
from pyneuromatic.core.nm_dataseries import NMDataSeriesContainer
from pyneuromatic.core.nm_notes import NMNotes
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
from pyneuromatic.analysis.nm_tool_folder import NMToolFolderContainer
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_utilities as nmu


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
    def toolfolder(self) -> NMToolFolderContainer:
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

    # DataSeries creation from data names

    def detect_prefixes(self) -> list[str]:
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
        :meth:`assemble_dataseries` which creates the dataseries if it does not
        yet exist, or extends it with any new channels and epochs if it does.

        Use ``ch_start`` and ``ep_start`` to append to an existing dataseries::

            folder.new_dataseries("Record", n_channels=2, n_epochs=5)
            # Later, append 3 more epochs to both channels:
            folder.new_dataseries("Record", n_channels=2, n_epochs=3, ep_start=5)

        Args:
            prefix: Wave prefix (e.g. ``"Record"``). Must satisfy
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

        # Pre-flight: check data names don't conflict before touching any state
        for ch_i in range(ch_start, ch_start + n_channels):
            ch_char = nmu.channel_char(ch_i)
            for ep in range(ep_start, ep_start + n_epochs):
                name = "%s%s%d" % (prefix, ch_char, ep)
                if name in self.data:
                    raise ValueError("data %r already exists" % name)

        xscale = {
            "start": x_start, "delta": dx, "label": x_label, "units": x_units,
        }
        yscale = {"label": y_label, "units": y_units}

        for ch_i in range(ch_start, ch_start + n_channels):
            ch_char = nmu.channel_char(ch_i)
            for ep in range(ep_start, ep_start + n_epochs):
                name = "%s%s%d" % (prefix, ch_char, ep)
                arr = fill(n_points) if callable(fill) else np.full(n_points, fill)
                self.data.new(
                    name=name, nparray=arr, xscale=xscale, yscale=yscale,
                    quiet=True,
                )

        return self.assemble_dataseries(prefix, select=select, quiet=quiet)

    def assemble_dataseries(
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

        Args:
            prefix: Prefix to match (case-insensitive). Can be partial.
            select: Whether to select the dataseries after creation/update.
            quiet: If True, suppress history output.

        Returns:
            The NMDataSeries (new or updated), or None if no matching data found.

        Example:
            >>> folder.assemble_dataseries("Record")
            # Creates or updates dataseries "Record" from RecordA0, RecordB0 …
        """
        from pyneuromatic.core.nm_data import NMData

        if not prefix or not isinstance(prefix, str):
            return None

        # Find ALL data in the folder matching the prefix pattern.
        # Key: (channel_char, epoch_num), Value: NMData
        matches: dict[tuple[str, int], NMData] = {}
        actual_prefix: str | None = None

        for name, data in self.data.items():
            if not name.lower().startswith(prefix.lower()):
                continue
            parsed = nmu.parse_data_name(name)
            if parsed is None:
                continue
            detected_prefix, channel_char, epoch_num = parsed
            if not detected_prefix.lower().startswith(prefix.lower()):
                continue
            if actual_prefix is None:
                actual_prefix = detected_prefix
            elif actual_prefix != detected_prefix:
                continue
            matches[(channel_char, epoch_num)] = data

        if not matches or actual_prefix is None:
            return None

        # Get or create the dataseries
        is_new = actual_prefix not in self.dataseries
        if is_new:
            ds = self.dataseries.new(name=actual_prefix, select=select, quiet=True)
            if ds is None:
                return None
            channel_map: dict[str, object] = {}
            epoch_map: dict[int, object] = {}
        else:
            ds = self.dataseries.get(actual_prefix)
            if select:
                self.dataseries.selected_name = actual_prefix
            # Pre-populate maps from existing channels and epochs so we can
            # detect what is new vs already present
            channel_map = {
                ch_name: ds.channels.get(ch_name) for ch_name in ds.channels
            }
            epoch_map = {}
            for ep_name in ds.epochs:
                try:
                    epoch_map[int(ep_name[1:])] = ds.epochs.get(ep_name)
                except (ValueError, IndexError):
                    pass

        # Determine which channels/epochs appear in the matches
        channel_chars = sorted(set(ch for ch, _ in matches.keys()))
        epoch_nums = sorted(set(ep for _, ep in matches.keys()))

        # Create only channels/epochs not already in the dataseries
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

        # Log summary
        ch_str = ", ".join(channel_chars)
        ep_str = ("E%d..E%d" % (epoch_nums[0], epoch_nums[-1])) if epoch_nums else ""
        action = "new" if is_new else "updated"
        nmh.history(
            "%s dataseries '%s': channels %s; epochs %s"
            % (action, actual_prefix, ch_str, ep_str),
            path=self.path_str,
            quiet=quiet,
        )

        return ds


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
