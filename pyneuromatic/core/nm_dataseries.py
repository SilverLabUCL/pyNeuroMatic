# -*- coding: utf-8 -*-
"""
NMDataSeries module for managing data series with channels and epochs.

A data series represents a collection of data where names follow the pattern:
RecordA0, RecordA1... where "Record" is the prefix, "A" is the channel,
and "0" is the epoch number.

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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyneuromatic.core.nm_data import NMData

from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
from pyneuromatic.core.nm_channel import NMChannel, NMChannelContainer
from pyneuromatic.core.nm_epoch import NMEpoch, NMEpochContainer
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_preferences as nmp
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
DataSeries:
      E0  E1  E2... (epochs)
Ch A  A0  A1  A2...
Ch B  B0  B1  B2...
.
.
.
"""


class NMDataSeries(NMObject):
    """
    NM DataSeries class.

    Coordinates channels and epochs for a data series. Data dimensions
    are managed by individual NMData objects (source of truth) and
    NMChannel objects (defaults/templates for new data).
    """

    # Extend NMObject's special attrs with NMDataSeries's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMDataSeries__channel_container",
        "_NMDataSeries__epoch_container",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMDataSeries0",  # dataseries name/prefix
    ) -> None:
        super().__init__(parent=parent, name=name)

        self.__channel_container: NMChannelContainer = NMChannelContainer(parent=self)
        self.__epoch_container: NMEpochContainer = NMEpochContainer(parent=self)

    # override
    def __eq__(
        self,
        other: object
    ) -> bool:
        if not isinstance(other, NMDataSeries):
            return NotImplemented
        if not super().__eq__(other):
            return False
        if self.__channel_container != other.__channel_container:
            return False
        if self.__epoch_container != other.__epoch_container:
            return False
        return True

    def __deepcopy__(self, memo: dict) -> NMDataSeries:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMDataSeries by bypassing __init__ and directly
        setting attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMDataSeries
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

        # Now handle NMDataSeries's special attributes

        # __channel_container: deep copy and update parent
        if isinstance(self._NMDataSeries__channel_container, NMChannelContainer):
            result._NMDataSeries__channel_container = copy.deepcopy(
                self._NMDataSeries__channel_container, memo
            )
            result._NMDataSeries__channel_container._parent = result
        else:
            result._NMDataSeries__channel_container = NMChannelContainer(parent=result)

        # __epoch_container: deep copy and update parent
        if isinstance(self._NMDataSeries__epoch_container, NMEpochContainer):
            result._NMDataSeries__epoch_container = copy.deepcopy(
                self._NMDataSeries__epoch_container, memo
            )
            result._NMDataSeries__epoch_container._parent = result
        else:
            result._NMDataSeries__epoch_container = NMEpochContainer(parent=result)

        return result

    # override
    @property
    def parameters(self) -> dict[str, object]:
        k = super().parameters
        return k

    # override
    @property
    def content(self) -> dict[str, str]:
        k = super().content
        k.update(self.channels.content)
        k.update(self.epochs.content)
        return k

    @property
    def channels(self) -> NMChannelContainer:
        return self.__channel_container

    @property
    def epochs(self) -> NMEpochContainer:
        return self.__epoch_container

    def get_selected(
        self,
        get_keys: bool = False
    ) -> list[NMData] | list[str]:
        """Get data at the intersection of selected channel and epoch.

        Returns:
            List of NMData objects (or their names if get_keys=True) that
            belong to both the selected channel and selected epoch.
        """
        if not self.channels.selected_name:
            return []
        c = self.channels.selected_value
        if c is None:
            return []
        if not self.epochs.selected_name:
            return []
        e = self.epochs.selected_value
        if e is None:
            return []

        dlist: list = []
        for d in e.data:
            if d in c.data:
                if get_keys:
                    dlist.append(d.name)
                else:
                    dlist.append(d)
        return dlist

    def get_data(
        self,
        channel: str | None = None,
        epoch: str | None = None
    ) -> NMData | None:
        """Get data at the intersection of a channel and epoch.

        Args:
            channel: Channel name (e.g., "A"). If None, uses selected channel.
            epoch: Epoch name (e.g., "E0"). If None, uses selected epoch.

        Returns:
            NMData object at the intersection, or None if not found.
        """
        if channel is None:
            c = self.channels.selected_value
        else:
            c = self.channels.get(channel)
        if c is None:
            return None

        if epoch is None:
            e = self.epochs.selected_value
        else:
            e = self.epochs.get(epoch)
        if e is None:
            return None

        for d in c.data:
            if d in e.data:
                return d
        return None

    # =========================================================================
    # Bulk scale utility methods
    #
    # These methods provide convenient ways to set scale properties across
    # multiple data objects. NMData remains the source of truth for scales.
    # =========================================================================

    def set_xstart(
        self,
        value: float | int,
        channel: str | None = None,
        quiet: bool = nmp.QUIET,
    ) -> int:
        """Set x-axis start value on data in this series.

        Args:
            value: The x-start value to set.
            channel: If specified, only set for this channel. If None, set for all.

        Returns:
            Number of data objects modified.
        """
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "value", "number"))

        count = 0
        for ch in self._iter_channels(channel):
            for data in ch.data:
                data.xscale._set_start(value, quiet=True)
                count += 1
        if count > 0:
            ch_str = channel if channel else "all"
            nmh.history(
                "set xstart=%s, channel=%s, n=%d" % (value, ch_str, count),
                path=self.path_str,
                quiet=quiet,
            )
        return count

    def set_xdelta(
        self,
        value: float | int,
        channel: str | None = None,
        quiet: bool = nmp.QUIET,
    ) -> int:
        """Set x-axis delta (sample interval) on data in this series.

        Args:
            value: The x-delta value to set.
            channel: If specified, only set for this channel. If None, set for all.

        Returns:
            Number of data objects modified.
        """
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "value", "number"))

        count = 0
        for ch in self._iter_channels(channel):
            for data in ch.data:
                data.xscale._set_delta(value, quiet=True)
                count += 1
        if count > 0:
            ch_str = channel if channel else "all"
            nmh.history(
                "set xdelta=%s, channel=%s, n=%d" % (value, ch_str, count),
                path=self.path_str,
                quiet=quiet,
            )
        return count

    def set_xlabel(
        self,
        label: str,
        channel: str | None = None,
        quiet: bool = nmp.QUIET,
    ) -> int:
        """Set x-axis label on data in this series.

        Args:
            label: The x-axis label to set.
            channel: If specified, only set for this channel. If None, set for all.

        Returns:
            Number of data objects modified.
        """
        if not isinstance(label, str):
            raise TypeError(nmu.type_error_str(label, "label", "string"))

        count = 0
        for ch in self._iter_channels(channel):
            for data in ch.data:
                data.xscale._set_label(label, quiet=True)
                count += 1
        if count > 0:
            ch_str = channel if channel else "all"
            nmh.history(
                "set xlabel='%s', channel=%s, n=%d" % (label, ch_str, count),
                path=self.path_str,
                quiet=quiet,
            )
        return count

    def set_xunits(
        self,
        units: str,
        channel: str | None = None,
        quiet: bool = nmp.QUIET,
    ) -> int:
        """Set x-axis units on data in this series.

        Args:
            units: The x-axis units to set (e.g., "ms", "s").
            channel: If specified, only set for this channel. If None, set for all.

        Returns:
            Number of data objects modified.
        """
        if not isinstance(units, str):
            raise TypeError(nmu.type_error_str(units, "units", "string"))

        count = 0
        for ch in self._iter_channels(channel):
            for data in ch.data:
                data.xscale._set_units(units, quiet=True)
                count += 1
        if count > 0:
            ch_str = channel if channel else "all"
            nmh.history(
                "set xunits='%s', channel=%s, n=%d" % (units, ch_str, count),
                path=self.path_str,
                quiet=quiet,
            )
        return count

    def set_ylabel(
        self,
        label: str,
        channel: str | None = None,
        quiet: bool = nmp.QUIET,
    ) -> int:
        """Set y-axis label on data in this series.

        Args:
            label: The y-axis label to set.
            channel: If specified, only set for this channel. If None, set for all.

        Returns:
            Number of data objects modified.
        """
        if not isinstance(label, str):
            raise TypeError(nmu.type_error_str(label, "label", "string"))

        count = 0
        for ch in self._iter_channels(channel):
            for data in ch.data:
                data.yscale._set_label(label, quiet=True)
                count += 1
        if count > 0:
            ch_str = channel if channel else "all"
            nmh.history(
                "set ylabel='%s', channel=%s, n=%d" % (label, ch_str, count),
                path=self.path_str,
                quiet=quiet,
            )
        return count

    def set_yunits(
        self,
        units: str,
        channel: str | None = None,
        quiet: bool = nmp.QUIET,
    ) -> int:
        """Set y-axis units on data in this series.

        Args:
            units: The y-axis units to set (e.g., "pA", "mV").
            channel: If specified, only set for this channel. If None, set for all.

        Returns:
            Number of data objects modified.
        """
        if not isinstance(units, str):
            raise TypeError(nmu.type_error_str(units, "units", "string"))

        count = 0
        for ch in self._iter_channels(channel):
            for data in ch.data:
                data.yscale._set_units(units, quiet=True)
                count += 1
        if count > 0:
            ch_str = channel if channel else "all"
            nmh.history(
                "set yunits='%s', channel=%s, n=%d" % (units, ch_str, count),
                path=self.path_str,
                quiet=quiet,
            )
        return count

    def get_xscales_summary(self) -> dict[str, dict]:
        """Get a summary of x-scale values across all channels.

        Useful for checking if all data in a channel share the same x-scale,
        or for diagnostics after processing operations.

        Returns:
            Dict mapping channel names to their x-scale summary:
            {
                "A": {"start": {0.0}, "delta": {0.01}, "uniform": True},
                "B": {"start": {0.0, 0.5}, "delta": {0.02}, "uniform": False},
            }
        """
        result: dict[str, dict] = {}
        for ch_name, channel in self.channels.items():
            if not isinstance(channel, NMChannel):
                continue
            starts: set = set()
            deltas: set = set()
            for data in channel.data:
                starts.add(data.xscale.start)
                deltas.add(data.xscale.delta)
            result[ch_name] = {
                "start": starts,
                "delta": deltas,
                "uniform": len(starts) <= 1 and len(deltas) <= 1
            }
        return result

    def get_yscales_summary(self) -> dict[str, dict]:
        """Get a summary of y-scale values across all channels.

        Returns:
            Dict mapping channel names to their y-scale summary:
            {
                "A": {"label": {"current"}, "units": {"pA"}},
                "B": {"label": {"voltage"}, "units": {"mV"}},
            }
        """
        result: dict[str, dict] = {}
        for ch_name, channel in self.channels.items():
            if not isinstance(channel, NMChannel):
                continue
            labels: set = set()
            units: set = set()
            for data in channel.data:
                if data.yscale.label:
                    labels.add(data.yscale.label)
                if data.yscale.units:
                    units.add(data.yscale.units)
            result[ch_name] = {
                "label": labels,
                "units": units,
            }
        return result

    def _iter_channels(self, channel: str | None = None):
        """Iterate over channels, optionally filtering to a single channel.

        Args:
            channel: If specified, yield only this channel. If None, yield all.

        Yields:
            NMChannel objects.

        Raises:
            KeyError: If specified channel does not exist.
        """
        if channel is not None:
            c = self.channels.get(channel)
            if c is None:
                raise KeyError(f"channel '{channel}' does not exist")
            if isinstance(c, NMChannel):
                yield c
        else:
            for ch in self.channels.values():
                if isinstance(ch, NMChannel):
                    yield ch


class NMDataSeriesContainer(NMObjectContainer):
    """
    Container of NMDataSeries
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMDataSeriesContainer0",
    ) -> None:
        super().__init__(
            parent=parent,
            name=name,
            rename_on=False,
            auto_name_prefix="",  # no prefix
            auto_name_seq_format="",
        )

    # override, no super
    def content_type(self) -> str:
        return NMDataSeries.__name__

    # override
    def new(
        self,
        name: str = "",  # dataseries name/prefix
        select: bool = False,
        # quiet: bool = nmp.QUIET
    ) -> NMDataSeries | None:
        name = self._newkey(name)
        # Use self._parent (NMFolder) to skip container in parent chain,
        # consistent with NMFolder, NMData, NMChannel, NMEpoch
        s = NMDataSeries(parent=self._parent, name=name)
        if super()._new(s, select=select):
            return s
        return None

    # override, no super
    def duplicate(self):
        raise RuntimeError("dataseries cannot be duplicated")
