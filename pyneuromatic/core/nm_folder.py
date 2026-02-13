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
import pyneuromatic.core.nm_utilities as nmu


"""
NM class tree:

NMManager
    NMProject (root)
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

    def toolresults_save(self, tool: str, results: Any) -> str:
        imax_keys = 99
        if not isinstance(tool, str):
            e = nmu.type_error_str(tool, "tool", "string")
            raise TypeError(e)

        tp = self.path_str
        foundkey = False
        for i in range(imax_keys):
            newkey = tool + str(i)
            if newkey not in self.__toolresults:
                foundkey = True
                break
        if not foundkey:
            e = "failed to find unused key for %s results in %s" % (tool, tp)
            raise KeyError(e)

        t = str(datetime.datetime.now())
        r = {}
        r["tool"] = tool
        r["date"] = t
        r["results"] = results
        self.__toolresults[newkey] = r
        print("saved %s results to %s via key '%s' (%s)" %
              (tool, tp, newkey, t))
        return newkey

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

    def make_dataseries(
        self,
        prefix: str,
        select: bool = False
    ) -> NMDataSeries | None:
        """Create a dataseries from data matching a prefix pattern.

        Scans the data container for names matching the NeuroMatic pattern
        {prefix}{channel}{epoch} and creates a dataseries with appropriate
        channels and epochs, linking the NMData objects.

        The prefix can be partial - for example, "Rec" will match "RecordA0".
        The full detected prefix (e.g., "Record") becomes the dataseries name.

        Args:
            prefix: Prefix to match (case-insensitive). Can be partial.
            select: Whether to select the new dataseries.

        Returns:
            The created NMDataSeries, or None if no matching data found.

        Example:
            >>> folder.make_dataseries("Record")
            # Creates dataseries "Record" with channels A, B and epochs E0, E1
            # if folder contains RecordA0, RecordA1, RecordB0, RecordB1
        """
        from pyneuromatic.core.nm_data import NMData
        from pyneuromatic.core.nm_dataseries import NMDataSeries

        if not prefix or not isinstance(prefix, str):
            return None

        # Find all data matching the prefix pattern
        # Key: (channel_char, epoch_num), Value: NMData
        matches: dict[tuple[str, int], NMData] = {}
        actual_prefix: str | None = None

        for name, data in self.data.items():
            # Check if name starts with user's prefix (case-insensitive)
            if not name.lower().startswith(prefix.lower()):
                continue

            # Parse the name from the end to find channel and epoch
            parsed = nmu.parse_data_name(name)
            if parsed is None:
                continue

            detected_prefix, channel_char, epoch_num = parsed

            # Verify detected prefix starts with user's prefix
            if not detected_prefix.lower().startswith(prefix.lower()):
                continue

            # Use first detected prefix as the actual prefix for dataseries name
            if actual_prefix is None:
                actual_prefix = detected_prefix
            elif actual_prefix != detected_prefix:
                # Different prefix detected, skip this data
                continue

            matches[(channel_char, epoch_num)] = data

        if not matches or actual_prefix is None:
            return None

        # Check if dataseries already exists
        if actual_prefix in self.dataseries:
            return None  # Could also raise an error or return existing

        # Create the dataseries
        ds = self.dataseries.new(name=actual_prefix, select=select)
        if ds is None:
            return None

        # Determine unique channels and epochs
        channel_chars = sorted(set(ch for ch, _ in matches.keys()))
        epoch_nums = sorted(set(ep for _, ep in matches.keys()))

        # Create channels (NMChannelContainer auto-names A, B, C...)
        # We need to create channels in order, so if data has A, B, C, we create 3
        channel_map: dict[str, object] = {}  # channel_char -> NMChannel
        for ch_char in channel_chars:
            channel = ds.channels.new()
            if channel is not None:
                channel_map[ch_char] = channel

        # Create epochs (NMEpochContainer auto-names E0, E1, E2...)
        epoch_map: dict[int, object] = {}  # epoch_num -> NMEpoch
        for ep_num in epoch_nums:
            epoch = ds.epochs.new()
            if epoch is not None:
                epoch_map[ep_num] = epoch

        # Link data to channels and epochs
        for (ch_char, ep_num), data in matches.items():
            channel = channel_map.get(ch_char)
            epoch = epoch_map.get(ep_num)

            if channel is not None:
                channel.data.append(data)
            if epoch is not None:
                epoch.data.append(data)

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
        # quiet: bool = nmp.QUIET
    ) -> NMFolder | None:
        actual_name = self._newkey(name)
        # Use self._parent (NMProject) to skip container in parent chain,
        # consistent with NMData, NMDataSeries, NMChannel, NMEpoch
        f = NMFolder(parent=self._parent, name=actual_name)
        if super()._new(f, select=select):
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
