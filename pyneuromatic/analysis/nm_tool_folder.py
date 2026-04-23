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

from pyneuromatic.core.nm_data import NMData, NMDataContainer
from pyneuromatic.core.nm_dataseries import NMDataSeries, NMDataSeriesContainer
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_utilities as nmu


class NMToolFolder(NMObject):
    """
    NM Data Folder class
    """

    # Extend NMObject's special attrs with NMToolFolder's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMToolFolder__data_container",
        "_NMToolFolder__dataseries_container",
        "_NMToolFolder__dataseries",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMToolFolder0",
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.__data_container: NMDataContainer = NMDataContainer(parent=self)
        self.__dataseries: NMDataSeriesContainer = NMDataSeriesContainer(parent=self)

    # override
    def __eq__(
        self,
        other: object
    ) -> bool:
        if not isinstance(other, NMToolFolder):
            return NotImplemented
        if not super().__eq__(other):
            return False
        if self.__data_container != other.data:
            return False
        return True

    def __deepcopy__(self, memo: dict) -> NMToolFolder:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMToolFolder by bypassing __init__ and directly
        setting attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMToolFolder
        """
        import datetime

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

        # Now handle NMToolFolder's special attributes

        # __data_container: deep copy and update parent
        if self._NMToolFolder__data_container is not None:
            result._NMToolFolder__data_container = copy.deepcopy(
                self._NMToolFolder__data_container, memo
            )
            result._NMToolFolder__data_container._parent = result
        else:
            result._NMToolFolder__data_container = NMDataContainer(parent=result)

        # __dataseries: deep copy and update parent
        if self._NMToolFolder__dataseries is not None:
            result._NMToolFolder__dataseries = copy.deepcopy(
                self._NMToolFolder__dataseries, memo
            )
            result._NMToolFolder__dataseries._parent = result
        else:
            result._NMToolFolder__dataseries = NMDataSeriesContainer(parent=result)

        return result

    # override
    @property
    def content(self) -> dict[str, str]:
        k = super().content
        if self.__data_container is not None:
            k.update(self.__data_container.content)
        return k

    @property
    def data(self) -> NMDataContainer | None:
        return self.__data_container

    @property
    def dataseries(self) -> NMDataSeriesContainer:
        return self.__dataseries

    def build_dataseries(
        self,
        prefix: str,
        matches: dict[tuple[str, int], NMData],
    ) -> NMDataSeries | None:
        """Build or extend an :class:`~pyneuromatic.core.nm_dataseries.NMDataSeries`
        from an explicit matches mapping.

        Creates the dataseries if it does not yet exist, then creates any
        channels and epochs not already present, and links each NMData to its
        channel and epoch.  Already-linked data is silently skipped so this
        method is safe to call on an existing dataseries.

        Modelled on :meth:`~pyneuromatic.core.nm_folder.NMFolder.build_dataseries`.

        Args:
            prefix:  Dataseries name (e.g. ``"SPK_"``).
            matches: Mapping of ``(channel_char, epoch_num) -> NMData``.
                Channel chars must be single uppercase letters (e.g. ``"A"``);
                epoch numbers are zero-based integers.

        Returns:
            The :class:`~pyneuromatic.core.nm_dataseries.NMDataSeries` (new or
            updated), or ``None`` if *matches* is empty.
        """
        if not matches:
            return None

        is_new = prefix not in self.__dataseries
        if is_new:
            ds = self.__dataseries.new(name=prefix, quiet=True)
            if ds is None:
                return None
            channel_map: dict[str, object] = {}
            epoch_map: dict[int, object] = {}
        else:
            ds = self.__dataseries.get(prefix)
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

        for ch_char in channel_chars:
            if ch_char not in channel_map:
                ch = ds.channels.new(quiet=True)
                if ch is not None:
                    channel_map[ch_char] = ch

        for ep_num in epoch_nums:
            if ep_num not in epoch_map:
                ep = ds.epochs.new(quiet=True)
                if ep is not None:
                    epoch_map[ep_num] = ep

        for (ch_char, ep_num), data in matches.items():
            ch = channel_map.get(ch_char)
            ep = epoch_map.get(ep_num)
            if ch is not None and data not in ch.data:
                ch.data.append(data)
            if ep is not None and data not in ep.data:
                ep.data.append(data)

        return ds


class NMToolFolderContainer(NMObjectContainer):
    """
    Container of NMToolFolders
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMToolFolderContainer0",
        rename_on: bool = True,
        name_prefix: str = "toolfolder",
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
        return NMToolFolder.__name__

    # override
    def new(
        self,
        name: str | None = None,
        select: bool = False,
        quiet: bool = nmc.QUIET
    ) -> NMToolFolder | None:
        name = self._newkey(name)
        f = NMToolFolder(parent=self, name=name)
        if super()._add(f, select=select, quiet=quiet):
            return f
        return None

    def get_or_create(
        self,
        base: str,
        overwrite: bool = True,
    ) -> NMToolFolder:
        """Return a tool subfolder for *base*, respecting the overwrite flag.

        Args:
            base: Base name without trailing sequence number, e.g.
                ``"spike_Record_A"`` or ``"stats_Record_A"``.
            overwrite: If ``True`` (default), target ``{base}_0``.  If that
                subfolder already exists, its data container is cleared and it
                is returned for reuse.  If it does not exist, it is created.
                If ``False``, find the first unused name ``{base}_0``,
                ``{base}_1``, … and create a new subfolder there.

        Returns:
            The target :class:`NMToolFolder`.
        """
        if overwrite:
            name = "%s_0" % base
            existing = self.get(name)
            if existing is not None:
                existing.data.clear(quiet=True)
                return existing
            return self.new(name=name)
        i = 0
        f = None
        while f is None:
            try:
                f = self.new(name="%s_%d" % (base, i))
            except KeyError:
                i += 1
        return f