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
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu


NP_ORDER = "C"
NP_DTYPE = numpy.float64
NP_FILL_VALUE = numpy.nan

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


class NMData(NMObject):
    """
    NM Data class.

    Holds a numpy array of data (y-values), optional x-array,
    and x/y scale metadata as simple dicts.
    """

    # Extend NMObject's special attrs with NMData's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMData__nparray",
        "_NMData__xarray",
        "_NMData__dataseries_channel",
        "_NMData__dataseries_epoch",
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

        # X-scale dict
        if xscale is None:
            self.__xscale: dict = {"label": "", "units": "", "start": 0, "delta": 1}
        elif isinstance(xscale, dict):
            self.__xscale = dict(xscale)
        else:
            e = nmu.type_error_str(xscale, "xscale", "dictionary")
            raise TypeError(e)

        # Y-scale dict
        if yscale is None:
            self.__yscale: dict = {"label": "", "units": ""}
        elif isinstance(yscale, dict):
            self.__yscale = dict(yscale)
        else:
            e = nmu.type_error_str(yscale, "yscale", "dictionary")
            raise TypeError(e)

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
        if not _eq_arrays(self.__xarray, other.xarray):
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
        # (this handles __xscale and __yscale dicts automatically)
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
        k.update({"xscale": dict(self.xscale)})
        k.update({"yscale": dict(self.yscale)})
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
        return self.__xarray

    @xarray.setter
    def xarray(self, xarray: numpy.ndarray | None) -> None:
        if xarray is not None:
            if not isinstance(xarray, numpy.ndarray):
                e = nmu.type_error_str(xarray, "xarray", "numpy.ndarray")
                raise TypeError(e)
        self.__xarray = xarray

    # =========================================================================
    # Scale properties (delegate to channel when dataseries-linked)
    # =========================================================================

    @property
    def xscale(self) -> dict:
        if isinstance(self.__dataseries_channel, NMChannel):
            return self.__dataseries_channel.xscale
        return self.__xscale

    @property
    def yscale(self) -> dict:
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
        """Convert an x-value to an array index.

        Args:
            xvalue: The x-axis value to find.
            clip: If True, clip out-of-bounds values to array limits.

        Returns:
            The corresponding array index, or None if not found.
        """
        if not (isinstance(xvalue, (float, int)) and not isinstance(xvalue, bool)):
            e = nmu.type_error_str(xvalue, "xvalue", "float")
            raise TypeError(e)

        if isinstance(self.__xarray, numpy.ndarray):
            if math.isinf(xvalue):
                if xvalue < 0:
                    return 0
                else:
                    return self.__xarray.size - 1
            indexes = numpy.argwhere(self.__xarray >= xvalue)
            shape = indexes.shape  # (N, 1)
            if len(shape) != 2:
                return None
            if shape[0] > 0:
                return indexes[0][0]
            return None

        # Use start/delta from xscale
        xscale = self.xscale
        start = xscale.get("start")
        delta = xscale.get("delta")

        if start is None or delta is None:
            return None

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
        """Convert an array index to an x-value.

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
        if isinstance(self.__xarray, numpy.ndarray):
            points = self.__xarray.size
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

        if isinstance(self.__xarray, numpy.ndarray):
            return self.__xarray[i]

        xscale = self.xscale
        start = xscale.get("start")
        delta = xscale.get("delta")

        if start is None or delta is None:
            return None

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
    if nmp.NAN_EQ_NAN:
        if numpy.allclose(a1, a2, rtol=0, atol=0, equal_nan=True):
            return True
    return False


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

    # override, no super
    def content_type(self) -> str:
        return NMData.__name__

    # override
    def new(
        self,
        name: str | None = None,
        select: bool = False,
        nparray: numpy.ndarray | None = None,
        xscale: dict | None = None,
        yscale: dict | None = None,
        # quiet: bool = nmp.QUIET
    ) -> NMData | None:
        actual_name = self._newkey(name)
        # Use self._parent (NMFolder) to skip container in parent chain,
        # consistent with NMFolder, NMDataSeries, NMChannel, NMEpoch
        d = NMData(
            parent=self._parent,
            name=actual_name,
            nparray=nparray,
            xscale=xscale,
            yscale=yscale,
        )
        if super()._new(d, select=select):
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
