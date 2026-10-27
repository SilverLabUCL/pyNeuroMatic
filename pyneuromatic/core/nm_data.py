# -*- coding: utf-8 -*-
"""
NMData module.

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
import math
import numpy

from pyneuromatic.core.nm_channel import NMChannel
from pyneuromatic.core.nm_dataseries import NMDataSeries, NMDataSeriesContainer
from pyneuromatic.core.nm_epoch import NMEpoch
from pyneuromatic.core.nm_notes import NMNotes
from pyneuromatic.core.nm_object import NMObject
from pyneuromatic.core.nm_object_container import NMObjectContainer
from pyneuromatic.core.nm_scale import NMScaleX, NMScaleY, _xscale_from_dict, _yscale_from_dict
import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_utilities as nmu


NP_ORDER = "C"
NP_DTYPE = numpy.float64
NP_FILL_VALUE = numpy.nan

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


class NMData(NMObject):
    """
    NM Data class.

    Holds a numpy array of data (yvalues), optional x-array,
    and x/y scale metadata as simple dicts.
    """

    # Extend NMObject's special attrs with NMData's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMData__nparray",
        "_NMData__xarray",
        "_NMData__xarray_source",
        "_NMData__xscale",
        "_NMData__yscale",
        "_NMData__dataseries_channel",
        "_NMData__dataseries_epoch",
        # Set to None on a copy (its container links it): never copied, which
        # would copy every array of the folder to duplicate one
        "_container",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMData0",
        nparray: numpy.ndarray | None = None,
        xscale: dict | None = None,
        yscale: dict | None = None,
        xarray: numpy.ndarray | None = None,
        dataseries_channel: NMChannel | None = None,
        dataseries_epoch: NMEpoch | None = None,
    ) -> None:
        super().__init__(parent=parent, name=name)  # NMObject

        self.__dataseries_channel: NMChannel | None = None
        self.__dataseries_epoch: NMEpoch | None = None

        # Y-data array
        if nparray is not None:
            if not isinstance(nparray, numpy.ndarray):
                e = nmu.type_error_str(nparray, "nparray", "numpy.ndarray")
                raise TypeError(e)
        self.__nparray = nparray

        # Optional explicit x-data array
        if xarray is not None:
            if not isinstance(xarray, numpy.ndarray):
                e = nmu.type_error_str(xarray, "xarray", "numpy.ndarray")
                raise TypeError(e)
        self.__xarray = xarray
        # The array whose values are the x values (a reference, not a copy)
        self.__xarray_source: NMData | None = None

        # X-scale and Y-scale
        self.__xscale: NMScaleX = _xscale_from_dict(xscale, parent=self)
        self.__yscale: NMScaleY = _yscale_from_dict(yscale, parent=self)

        self.__notes = NMNotes()

        self._dataseries_set(dataseries_channel, dataseries_epoch)

    # override
    def __eq__(
        self,
        other: object,
    ) -> bool:
        if not isinstance(other, NMData):
            return NotImplemented
        if not super().__eq__(other):
            return False

        if self.xscale != other.xscale:
            return False
        if self.yscale != other.yscale:
            return False
        if not _eq_arrays(self.__nparray, other.nparray):
            return False
        if not _eq_arrays(self.xarray, other.xarray):
            return False
        if self._dataseries_channel != other._dataseries_channel:
            return False
        if self._dataseries_epoch != other._dataseries_epoch:
            return False
        if self.__notes != other.__notes:
            return False

        return True

    def __deepcopy__(self, memo: dict) -> NMData:
        """Support Python's copy.deepcopy() protocol."""
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

        # Now handle NMData's special attributes

        # __xscale/__yscale: deep copy and reset parent
        result._NMData__xscale = copy.deepcopy(self._NMData__xscale, memo)
        result._NMData__xscale._parent = result
        result._NMData__yscale = copy.deepcopy(self._NMData__yscale, memo)
        result._NMData__yscale._parent = result

        # __nparray: deep copy numpy array (if present)
        if self._NMData__nparray is not None:
            result._NMData__nparray = self._NMData__nparray.copy()
        else:
            result._NMData__nparray = None

        # __xarray: deep copy numpy array (if present)
        if self._NMData__xarray is not None:
            result._NMData__xarray = self._NMData__xarray.copy()
        else:
            result._NMData__xarray = None

        # __xarray_source: the copied source if it is copied too (e.g. the
        # whole folder: fixed up by NMFolder once all its data is copied),
        # else the same array
        source = getattr(self, "_NMData__xarray_source", None)
        result._NMData__xarray_source = memo.get(id(source), source)
        if source is not None and id(source) not in memo:
            memo.setdefault(XARRAY_LINKS, []).append((result, source))

        # __dataseries_channel and __dataseries_epoch: copy references
        # If copying within a folder context, try to resolve to copied objects
        if result._folder is not None:
            # Try to resolve channel reference
            if self._NMData__dataseries_channel is not None:
                if id(self._NMData__dataseries_channel) in memo:
                    result._NMData__dataseries_channel = memo[id(self._NMData__dataseries_channel)]
                else:
                    result._NMData__dataseries_channel = self._NMData__dataseries_channel
            else:
                result._NMData__dataseries_channel = None

            # Try to resolve epoch reference
            if self._NMData__dataseries_epoch is not None:
                if id(self._NMData__dataseries_epoch) in memo:
                    result._NMData__dataseries_epoch = memo[id(self._NMData__dataseries_epoch)]
                else:
                    result._NMData__dataseries_epoch = self._NMData__dataseries_epoch
            else:
                result._NMData__dataseries_epoch = None
        else:
            # Direct copy: keep references
            result._NMData__dataseries_channel = self._NMData__dataseries_channel
            result._NMData__dataseries_epoch = self._NMData__dataseries_epoch

        return result

    # override
    @property
    def parameters(self) -> dict[str, object]:
        k = super().parameters
        k.update({"xscale": self.xscale.to_dict()})
        k.update({"yscale": self.yscale.to_dict()})
        if isinstance(self.__nparray, numpy.ndarray):
            k.update({"nparray": self.__nparray.dtype})
        else:
            k.update({"nparray": None})
        ds = self._dataseries
        if isinstance(ds, NMDataSeries):
            k.update({"dataseries": ds.name})
        else:
            k.update({"dataseries": None})
        if isinstance(self.__dataseries_channel, NMChannel):
            k.update({"dataseries channel": self.__dataseries_channel.name})
        else:
            k.update({"dataseries channel": None})
        if isinstance(self.__dataseries_epoch, NMEpoch):
            k.update({"dataseries epoch": self.__dataseries_epoch.name})
        else:
            k.update({"dataseries epoch": None})
        return k

    @property
    def notes(self) -> NMNotes:
        """Return notes for this data."""
        return self.__notes

    # =========================================================================
    # Data array properties
    # =========================================================================

    @property
    def nparray(self) -> numpy.ndarray | None:
        return self.__nparray

    @nparray.setter
    def nparray(self, nparray: numpy.ndarray | None) -> None:
        if nparray is not None:
            if not isinstance(nparray, numpy.ndarray):
                e = nmu.type_error_str(nparray, "nparray", "numpy.ndarray")
                raise TypeError(e)
        self.__nparray = nparray

    @property
    def xarray(self) -> numpy.ndarray | None:
        """The x values, if not from the x scale's start and delta: the
        values of the x array (:attr:`xarray_source`), or values set
        directly. None if the x array no longer fits (e.g. its number of
        points changed): start and delta are used (see :meth:`xarray_problem`)."""
        source = getattr(self, "_NMData__xarray_source", None)
        if source is not None:
            values = source.nparray
            if (isinstance(values, numpy.ndarray) and values.ndim == 1
                    and values.dtype.kind in "biuf"
                    and (not isinstance(self.__nparray, numpy.ndarray)
                         or values.size == self.__nparray.size)):
                return values
            return None
        return self.__xarray

    @xarray.setter
    def xarray(self, xarray: numpy.ndarray | None) -> None:
        if xarray is not None:
            if not isinstance(xarray, numpy.ndarray):
                e = nmu.type_error_str(xarray, "xarray", "numpy.ndarray")
                raise TypeError(e)
        self.__xarray = xarray
        self.__xarray_source = None  # values set directly

    @property
    def xarray_source(self) -> NMData | None:
        """The array whose values are the x values (see :meth:`set_xarray_from`),
        or None."""
        return getattr(self, "_NMData__xarray_source", None)

    @property
    def xarray_name(self) -> str | None:
        """The name of :attr:`xarray_source`, or None."""
        source = self.xarray_source
        return source.name if source is not None else None

    def set_xarray_from(self, source: NMData | None) -> None:
        """Use *source*'s values as the x values: a reference, so changing
        *source* changes them (deleting it is refused while in use, see
        :meth:`NMDataContainer.pop`). None goes back to the x scale's start
        and delta.

        Raises:
            TypeError: If *source* is not an NMData.
            ValueError: If *source* is this array, has no numeric values, a
                different number of points, or values that don't increase.
        """
        if source is None:
            self.xarray = None
            return
        if source is self:
            raise ValueError("'%s' can't be its own x array" % self.name)
        check_xarray(source, self)
        self.__xarray = None
        self.__xarray_source = source

    def xarray_problem(self) -> str | None:
        """Why the x array doesn't give the x values (e.g. it no longer has
        as many points), or None."""
        source = self.xarray_source
        if source is None:
            return None
        try:
            check_xarray(source, self)
        except (TypeError, ValueError) as error:
            return str(error)
        return None

    # =========================================================================
    # Scale properties (delegate to channel when dataseries-linked)
    # =========================================================================

    @property
    def xscale(self) -> NMScaleX:
        if isinstance(self.__dataseries_channel, NMChannel):
            return self.__dataseries_channel.xscale
        return self.__xscale

    @property
    def yscale(self) -> NMScaleY:
        if isinstance(self.__dataseries_channel, NMChannel):
            return self.__dataseries_channel.yscale
        return self.__yscale

    # =========================================================================
    # X-scale computation methods
    # =========================================================================

    def get_xindex(
        self,
        xvalue: float,
        clip: bool = False
    ) -> int | None:
        """Convert an xvalue to an array index.

        Args:
            xvalue: The x-axis value to find.
            clip: If True, clip out-of-bounds values to array limits.

        Returns:
            The corresponding array index, or None if not found.
        """
        if not (isinstance(xvalue, (float, int)) and not isinstance(xvalue, bool)):
            e = nmu.type_error_str(xvalue, "xvalue", "float")
            raise TypeError(e)

        xarray = self.xarray  # an x array's values, or values set directly
        if isinstance(xarray, numpy.ndarray):
            if math.isinf(xvalue):
                if xvalue < 0:
                    return 0
                else:
                    return xarray.size - 1
            indexes = numpy.argwhere(xarray >= xvalue)
            shape = indexes.shape  # (N, 1)
            if len(shape) != 2:
                return None
            if shape[0] > 0:
                return indexes[0][0]
            return None

        # Use start/delta from xscale
        xscale = self.xscale
        start = xscale.start
        delta = xscale.delta

        points = self.__nparray.size if self.__nparray is not None else None
        if points is None:
            return None

        if math.isinf(xvalue):
            if xvalue < 0:
                return 0
            else:
                return points - 1

        index = round((xvalue - start) / delta)

        if clip:
            index = max(index, 0)
            index = min(index, points - 1)
            return int(index)
        else:
            if 0 <= index < points:
                return int(index)
            else:
                return None

    def get_xvalue(
        self,
        index: int,
        clip: bool = False
    ) -> float | None:
        """Convert an array index to an xvalue.

        Args:
            index: The array index.
            clip: If True, clip out-of-bounds indices to array limits.

        Returns:
            The corresponding x-axis value, or None if not computable.
        """
        if isinstance(index, int) and not isinstance(index, bool):
            i = index
        elif isinstance(index, numpy.integer):
            i = int(index)
        elif isinstance(index, float):
            i = int(index)
        else:
            e = nmu.type_error_str(index, "index", "integer")
            raise TypeError(e)

        if i < 0:
            if clip:
                i = 0
            else:
                raise ValueError("negative index: %s" % index)

        # Determine points
        xarray = self.xarray  # an x array's values, or values set directly
        if isinstance(xarray, numpy.ndarray):
            points = xarray.size
        elif self.__nparray is not None:
            points = self.__nparray.size
        else:
            points = None

        if points is None:
            return None

        if i >= points:
            if clip:
                i = points - 1
            else:
                e = "index out of range: %s >= %s" % (index, points)
                raise ValueError(e)

        if isinstance(xarray, numpy.ndarray):
            return xarray[i]

        xscale = self.xscale
        start = xscale.start
        delta = xscale.delta

        return start + i * delta

    # =========================================================================
    # DataSeries properties
    # =========================================================================

    @property
    def _dataseries(self) -> NMDataSeries | None:
        if self.__dataseries_channel is None:
            return None
        if not isinstance(self.__dataseries_channel, NMChannel):
            return None
        if self.__dataseries_epoch is None:
            return None
        if not isinstance(self.__dataseries_epoch, NMEpoch):
            return None
        dataseries_c = self.__dataseries_channel._parent
        if not isinstance(dataseries_c, NMDataSeries):
            return None
        dataseries_e = self.__dataseries_epoch._parent
        if not isinstance(dataseries_e, NMDataSeries):
            return None
        if dataseries_c is dataseries_e:
            return dataseries_c
        else:
            e = (
                "data channel and epoch are not from the same dataseries: "
                "'%s' vs '%s'" % (dataseries_c.name, dataseries_e.name)
            )
            raise ValueError(e)

    @property
    def _dataseries_channel(self) -> NMChannel | None:
        return self.__dataseries_channel

    @property
    def _dataseries_epoch(self) -> NMEpoch | None:
        return self.__dataseries_epoch

    def _dataseries_set(
        self,
        channel: NMChannel | None,
        epoch: NMEpoch | None,
    ) -> NMDataSeries | None:
        if channel is None or isinstance(channel, NMChannel):
            self.__dataseries_channel = channel
        else:
            e = nmu.type_error_str(channel, "channel", "NMChannel")
            raise TypeError(e)
        if epoch is None or isinstance(epoch, NMEpoch):
            self.__dataseries_epoch = epoch
        else:
            e = nmu.type_error_str(epoch, "epoch", "NMEpoch")
            raise TypeError(e)
        return self._dataseries


def _eq_arrays(a1, a2) -> bool:
    """Compare two numpy arrays (or None) for equality."""
    if a1 is None and a2 is None:
        return True
    if a1 is None or a2 is None:
        return False
    if not isinstance(a1, numpy.ndarray) or not isinstance(a2, numpy.ndarray):
        return False
    if a1.dtype != a2.dtype:
        return False
    if a1.shape != a2.shape:
        return False
    if numpy.array_equal(a1, a2):
        return True
    # array_equal returns false if both arrays filled with NANs
    if nmc.NAN_EQ_NAN:
        if numpy.allclose(a1, a2, rtol=0, atol=0, equal_nan=True):
            return True
    return False


# Deep copies: (copied data, its x array) still to point to the copied x array
XARRAY_LINKS = "nm_xarray_links"


def check_xarray(source: NMData, data: NMData | None = None) -> numpy.ndarray:
    """Check *source*'s values can be x values: numeric, one dimension,
    no NaN, increasing, and (with *data*) as many points as *data*. Returns
    them (not a copy).

    Raises:
        TypeError: If *source* is not an NMData.
        ValueError: Saying what is wrong.
    """
    if not isinstance(source, NMData):
        raise TypeError(nmu.type_error_str(source, "source", "NMData"))
    values = source.nparray
    if not isinstance(values, numpy.ndarray) or values.ndim != 1 or values.size == 0:
        raise ValueError("x array '%s' has no values" % source.name)
    if values.dtype.kind not in "biuf":
        raise ValueError("x array '%s' is not numeric" % source.name)
    if data is not None and isinstance(data.nparray, numpy.ndarray) \
            and data.nparray.size != values.size:
        raise ValueError(
            "x array '%s' has %d points, but '%s' has %d"
            % (source.name, values.size, data.name, data.nparray.size)
        )
    as_float = values.astype(float, copy=False)
    if numpy.isnan(as_float).any():
        raise ValueError("x array '%s' has NaN values" % source.name)
    if values.size > 1 and not (numpy.diff(as_float) > 0).all():
        raise ValueError("x array '%s' must increase from point to point" % source.name)
    return values


class NMDataContainer(NMObjectContainer):
    """
    Container of NMData
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMDataContainer0",
        rename_on: bool = True,
        name_prefix: str = "data",
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
    def _nm_cmd_path(self) -> str | None:
        from pyneuromatic.core.nm_folder import NMFolder
        if isinstance(self._parent, NMFolder):
            return '%s.data' % self._parent._nm_cmd_path
        return None

    # override, no super
    def content_type(self) -> str:
        return NMData.__name__

    # --- data series membership ----------------------------------------------

    def dataseries_of(self, name: str) -> list[NMDataSeries]:
        """Data series of the parent folder that contain data *name*."""
        key = self._getkey(name)
        if key is None:
            raise KeyError("key name '%s' does not exist" % name)
        data = self[key]
        container = getattr(self._parent, "dataseries", None)
        if not isinstance(container, NMDataSeriesContainer):
            return []
        # Identity, not ==: NMData.__eq__ compares contents and is slow
        return [
            ds for ds in container.values()
            if any(d is data for channel in ds.channels.values() for d in channel.data)
        ]

    def locked_dataseries_for_name(self, name: str) -> NMDataSeries | None:
        """The locked data series whose naming pattern *name* fits, if any.

        Such a name would add data to the series, which a lock prevents.
        """
        parsed = nmu.parse_data_name(name) if isinstance(name, str) else None
        container = getattr(self._parent, "dataseries", None)
        if parsed is None or not isinstance(container, NMDataSeriesContainer):
            return None
        key = container._getkey(parsed[0])
        if key is None or not container[key].locked:
            return None
        return container[key]

    def _folders(self) -> list:
        """This folder, and the others of its manager (x arrays may be used
        across folders, e.g. after Copy to Folder)."""
        folder = self._parent
        folders = getattr(getattr(folder, "_container", None), "values", None)
        return list(folders()) if folders is not None else [folder]

    def xarray_users(self, name: str) -> list[NMData]:
        """Arrays whose x values are data *name*'s (its x array users)."""
        key = self._getkey(name)
        if key is None:
            raise KeyError("key name '%s' does not exist" % name)
        source = self[key]
        users = []
        for folder in self._folders():
            container = getattr(folder, "data", None)
            if container is None:
                continue
            users += [d for d in container.values()
                      if d is not source and d.xarray_source is source]
        return users

    def xarray_series(self, name: str) -> list[NMDataSeries]:
        """Data series whose arrays take their x values from data *name*."""
        users = {id(d) for d in self.xarray_users(name)}
        found = []
        for folder in self._folders():
            container = getattr(folder, "dataseries", None)
            if not isinstance(container, NMDataSeriesContainer):
                continue
            found += [ds for ds in container.values()
                      if any(id(d) in users for c in ds.channels.values() for d in c.data)]
        return found

    def xarray_in_use_text(self, name: str) -> str | None:
        """Why data *name* can't be deleted (it gives x values), or None."""
        users = self.xarray_users(name)
        if not users:
            return None
        series = self.xarray_series(name)
        if series:
            names = ", ".join("'%s'" % ds.name for ds in series)
            return ("'%s' is the x array of data series %s and cannot be deleted "
                    "while in use" % (name, names))
        names = ", ".join("'%s'" % d.name for d in users[:5])
        return "'%s' is the x array of %s and cannot be deleted while in use" % (name, names)

    def locked_xarray_series(self, name: str) -> NMDataSeries | None:
        """A locked data series whose x array is data *name*: its values
        are then protected, as the series' own are."""
        return next((ds for ds in self.xarray_series(name) if ds.locked), None)

    def _check_unlocked(self, key: str) -> None:
        for ds in self.dataseries_of(key):
            if ds.locked:
                raise RuntimeError(
                    "'%s' is part of locked data series '%s'" % (key, ds.name)
                )

    def _unlink_from_dataseries(self, key: str) -> list[NMDataSeries]:
        """Remove data *key* from the channels and epochs of its data series."""
        data = self[key]
        series = self.dataseries_of(key)
        for ds in series:
            for item in list(ds.channels.values()) + list(ds.epochs.values()):
                for index, member in enumerate(item.data):
                    if member is data:
                        del item.data[index]
                        break
        return series

    # override
    def pop(  # type: ignore[override]
        self,
        key: str,
        default: object = NMObjectContainer._POPDEFAULT,
        quiet: bool = nmc.QUIET,
    ) -> NMData | None:
        """Remove data *key*, also unlinking it from its data series.

        Raises:
            RuntimeError: If the data is part of a locked data series, or
                other arrays take their x values from it.
        """
        actual_key = self._getkey(key)
        if actual_key is not None:
            self._check_unlocked(actual_key)
            in_use = self.xarray_in_use_text(actual_key)
            if in_use:
                raise RuntimeError(in_use)
            self._unlink_from_dataseries(actual_key)
        return super().pop(key, default, quiet=quiet)

    # override
    def rename(
        self,
        name: str,
        newname: str | None = None,
        quiet: bool = nmc.QUIET,
    ) -> bool:
        """Rename data *name*; its data series are then resynced.

        The data leaves its series, and rejoins one if the new name still
        matches the series' naming pattern (e.g. ``RecordA0`` -> ``RecordA9``).

        Raises:
            RuntimeError: If the data is part of a locked data series, or the
                new name fits a locked series' naming pattern.
        """
        key = self._getkey(name)
        if key is None:
            return super().rename(name, newname, quiet=quiet)  # raises
        self._check_unlocked(key)
        if newname is not None:
            locked = self.locked_dataseries_for_name(newname)
            if locked is not None:
                raise RuntimeError(
                    "'%s' fits locked data series '%s'" % (newname, locked.name)
                )
        data = self[key]
        prefixes = [ds.name for ds in self._unlink_from_dataseries(key)]
        result = super().rename(name, newname, quiet=quiet)
        series = getattr(self._parent, "dataseries", None)
        sync = getattr(self._parent, "sync_dataseries", None)
        if callable(sync) and isinstance(series, NMDataSeriesContainer):
            # Also rejoin an existing series whose pattern the new name fits
            parsed = nmu.parse_data_name(data.name)
            if parsed is not None and parsed[0] in series and parsed[0] not in prefixes:
                prefixes.append(parsed[0])
            for prefix in prefixes:
                if prefix in series:
                    sync(prefix, quiet=True)
        return result

    def copy_to(
        self,
        name: str,
        container: NMDataContainer,
        newname: str | None = None,
        quiet: bool = nmc.QUIET,
    ) -> NMData:
        """Copy data *name* into another data *container* (e.g. another folder).

        The copy is named *newname*, or keeps its name. It is not linked to
        any data series.
        """
        key = self._getkey(name)
        if key is None:
            raise KeyError("key name '%s' does not exist" % name)
        if not isinstance(container, NMDataContainer):
            raise TypeError(nmu.type_error_str(container, "container", "NMDataContainer"))
        copy = self[key].copy()
        copy._dataseries_set(None, None)
        copy._parent = container._parent
        copy._name_set(newname=container._newkey(newname or key), quiet=True)
        container._add(copy, quiet=quiet)
        return copy

    # override
    def new(
        self,
        name: str | None = None,
        select: bool = False,
        nparray: numpy.ndarray | None = None,
        xarray: numpy.ndarray | None = None,
        xscale: dict | None = None,
        yscale: dict | None = None,
        quiet: bool = nmc.QUIET
    ) -> NMData | None:
        actual_name = self._newkey(name)
        # Use self._parent (NMFolder) to skip container in parent chain,
        # consistent with NMFolder, NMDataSeries, NMChannel, NMEpoch
        d = NMData(
            parent=self._parent,
            name=actual_name,
            nparray=nparray,
            xarray=xarray,
            xscale=xscale,
            yscale=yscale,
        )
        if super()._add(d, select=select, quiet=quiet):
            return d
        return None

    # override
    # TODO: this no longer works
    def remove(self, names=[], indexes=[], confirm=True, quiet=False):
        rlist = super().remove(
            names=names, indexes=indexes, confirm=confirm, quiet=quiet
        )
        dsc = self.dataseries
        if not dsc or not isinstance(dsc, NMDataSeriesContainer):
            return rlist
        for d in rlist:  # remove data refs from data series and sets
            for i in range(0, dsc.count):
                ds = dsc.getitem(index=i)
                if not ds or not ds.thedata:
                    continue
                for c, cdata in ds.thedata.items():
                    if d in cdata:
                        cdata.remove(d)
                for j in range(0, ds.sets.count):
                    s = ds.sets.getitem(index=j)
                    if d in s.theset:
                        s.discard(d)
        return rlist
