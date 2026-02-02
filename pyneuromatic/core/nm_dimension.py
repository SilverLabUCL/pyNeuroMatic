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
import math
import numpy as np

from pyneuromatic.core.nm_object import NMObject
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu


class NMDimension(NMObject):
    """
    NM Scale class

    Contains data y-scale parameters
    """

    # Extend NMObject's special attrs with NMDimension's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMObject._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMDimension__nparray",
    })
    
    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMDimension0",
        nparray=None,  # 1D TODO: typing
        scale: dict | None = None,  # "label" and "units"
    ) -> None:
        super().__init__(parent=parent, name=name)

        if nparray is None:
            pass
        elif isinstance(nparray, np.ndarray):
            if nparray.ndim != 1:
                e = ("NumPy array should have 1 dimension, not %s" %
                     nparray.ndim)
                raise ValueError(e)
        else:
            e = nmu.type_error_str(nparray, "nparray", "np.ndarray")
            raise TypeError(e)

        self.__nparray = nparray
        self.__label: str | None = None
        self.__units: str | None = None

        if scale is None:
            pass  # ok
        elif isinstance(scale, dict):
            self._scale_set(scale, quiet=True)
        else:
            e = nmu.type_error_str(scale, "scale", "dictionary")
            raise TypeError(e)

    # override
    def __eq__(
        self,
        other: object
    ) -> bool:
        if not isinstance(other, NMDimension):
            return NotImplemented
        if not super().__eq__(other):
            return False
        label1 = self.__label.lower() if self.__label else None
        label2 = other.label.lower() if other.label else None
        if label1 != label2:
            return False
        units1 = self.__units.lower() if self.__units else None
        units2 = other.units.lower() if other.units else None
        if units1 != units2:
            return False
        if not NMDimension.__eq_arrays(self.__nparray, other.nparray):
            return False
        return True

    def __deepcopy__(self, memo: dict) -> NMDimension:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMDimension by bypassing __init__ and directly
        setting attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMDimension
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
        result._NMObject__notes_on = self._NMObject__notes_on
        result._NMObject__notes = copy.deepcopy(self._NMObject__notes, memo)
        result._NMObject__rename_fxnref = result._name_set
        result._NMObject__copy_of = self

        # Now handle NMDimension's special attributes

        # __nparray: deep copy numpy array (if present)
        if self._NMDimension__nparray is not None:
            result._NMDimension__nparray = self._NMDimension__nparray.copy()
        else:
            result._NMDimension__nparray = None

        return result

    def __eq_arrays(a1, a2) -> bool:
        if a1 is None and a2 is None:
            return True
        if isinstance(a1, np.ndarray):
            if not isinstance(a2, np.ndarray):
                return False
            if a1.dtype != a2.dtype:
                return False
            if a1.shape != a2.shape:
                return False
            if a1.nbytes != a2.nbytes:
                return False
            if not np.array_equal(a1, a2):
                # array_equal returns false if both arrays filled with NANs
                if nmp.NAN_EQ_NAN:
                    # compare array elements within a tolerance
                    if not np.allclose(
                        a1,
                        a2,
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    ):
                        return False
                else:
                    return False
        return False

    # override
    @property
    def parameters(self) -> dict[str, object]:
        k = super().parameters
        if isinstance(self.__nparray, np.ndarray):
            k.update({"nparray": self.__nparray.dtype})
        else:
            k.update({"nparray": None})
        k.update(self.scale)
        return k

    @property
    def scale(self) -> dict[str, object]:
        return {"label": self.__label, "units": self.__units}

    @scale.setter
    def scale(self, scale: dict[str, object]) -> None:
        self._scale_set(scale)

    def _scale_set(
        self,
        scale: dict[str, object],
        quiet: bool = nmp.QUIET
    ) -> None:
        if not isinstance(scale, dict):
            e = nmu.type_error_str(scale, "scale", "dictionary")
            raise TypeError(e)
        for k, v in scale.items():
            k = k.lower()
            if k == "label":
                if isinstance(v, str) or v is None:
                    self._label_set(v, quiet=quiet)
            elif k == "units":
                if isinstance(v, str) or v is None:
                    self._units_set(v, quiet=quiet)
            else:
                raise KeyError("unknown key '%s'" % k)
        return None

    @property
    def label(self) -> str | None:
        return self.__label

    @label.setter
    def label(self, label: str | None) -> None:
        return self._label_set(label)

    def _label_set(
        self,
        label: str | None,
        quiet: bool = nmp.QUIET
    ) -> None:
        if label is not None and not isinstance(label, str):
            e = nmu.type_error_str(label, "label", "string")
            raise TypeError(e)
        if isinstance(label, str):
            label = label.strip()
        if label == self.__label:
            return None  # no change
        self.__label = label
        # h = nmh.history_change_str("label", old, label)
        # self.note = h
        # nmh.history(h, quiet=quiet)
        return None

    @property
    def units(self) -> str | None:
        return self.__units

    @units.setter
    def units(self, units: str | None) -> None:
        return self._units_set(units)

    def _units_set(
        self,
        units: str | None,
        quiet: bool = nmp.QUIET
    ) -> None:
        if units is not None and not isinstance(units, str):
            e = nmu.type_error_str(units, "units", "string")
            raise TypeError(e)
        if isinstance(units, str):
            units = units.strip()
        if units == self.__units:
            return None  # no change
        self.__units = units
        # h = nmh.history_change_str("units", old, units)
        # self.note = h
        # nmh.history(h, quiet=quiet)
        return None

    @property
    def nparray(self):
        return self.__nparray

    @nparray.setter
    def nparray(self, nparray) -> None:
        return self._nparray_set(nparray)

    def _nparray_set(
        self,
        nparray,
        # quiet=False
    ) -> None:
        if nparray is None:
            pass  # ok
        elif not isinstance(nparray, np.ndarray):
            e = nmu.type_error_str(nparray, "nparray", "np.ndarray")
            raise TypeError(e)
        self.__nparray = nparray
        return None


class NMDimensionX(NMDimension):
    """
    NM X-Scale class

    Contains data x-scale parameters
    """
    
    # Extend NMDimension's special attrs with NMDimensionX's own
    _DEEPCOPY_SPECIAL_ATTRS: frozenset[str] = NMDimension._DEEPCOPY_SPECIAL_ATTRS | frozenset({
        "_NMDimensionX__ypair",
    })

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMDimensionX0",
        nparray=None,  # TODO: typing
        ypair=None,  # TODO: typing
        scale: dict | None = None,
        # "label", "units", "start", "delta", "points"
    ) -> None:

        # declare parameters before calling super (so scale_set() is OK)
        self.__start: float | None = None
        self.__delta: float | None = None
        self.__points: int | None = None
        self.__ypair = None

        super().__init__(
            parent=parent,
            name=name,
            nparray=nparray,
            scale=scale,
        )

        self._ypair_set(ypair)

    # override
    def __eq__(
        self, 
        other: object
    ) -> bool:
        if not isinstance(other, NMDimensionX):
            return NotImplemented
        if not super().__eq__(other):
            return False
        if self.start != other.start:
            return False
        if self.delta != other.delta:
            return False
        if self.points != other.points:
            return False
        return True

    def __deepcopy__(self, memo: dict) -> NMDimensionX:
        """Support Python's copy.deepcopy() protocol.

        Creates a copy of this NMDimensionX by bypassing __init__ and directly
        setting attributes.

        Args:
            memo: Dictionary to track already copied objects (prevents cycles)

        Returns:
            A deep copy of this NMDimensionX
        """
        import datetime

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Use the class attribute for special attrs (includes parent's attrs)
        special_attrs = cls._DEEPCOPY_SPECIAL_ATTRS

        # Deep copy all attributes that aren't special
        for attr, value in self.__dict__.items():
            if attr not in special_attrs:
                setattr(result, attr, copy.deepcopy(value, memo))

        # Set NMObject's attributes with custom handling
        result._NMObject__created = datetime.datetime.now().isoformat(" ", "seconds")
        result._NMObject__parent = self._NMObject__parent
        result._NMObject__name = self._NMObject__name
        result._NMObject__notes_on = self._NMObject__notes_on
        result._NMObject__notes = copy.deepcopy(self._NMObject__notes, memo)
        result._NMObject__rename_fxnref = result._name_set
        result._NMObject__copy_of = self

        # Handle NMDimension's special attributes

        # __nparray: deep copy numpy array (if present)
        if self._NMDimension__nparray is not None:
            result._NMDimension__nparray = self._NMDimension__nparray.copy()
        else:
            result._NMDimension__nparray = None

        # Handle NMDimensionX's special attributes

        # __ypair: deep copy numpy array (if present)
        if self._NMDimensionX__ypair is not None:
            result._NMDimensionX__ypair = self._NMDimensionX__ypair.copy()
        else:
            result._NMDimensionX__ypair = None

        return result

    # override
    def _scale_set(
        self,
        scale: dict[str, object],
        quiet: bool = nmp.QUIET
    ) -> None:
        if not isinstance(scale, dict):
            e = nmu.type_error_str(scale, "scale", "dictionary")
            raise TypeError(e)
        for k, v in scale.items():
            k = k.lower()
            if k == "label":
                if isinstance(v, str) or v is None:
                    self._label_set(v, quiet=quiet)
            elif k == "units":
                if isinstance(v, str) or v is None:
                    self._units_set(v, quiet=quiet)
            elif k == "start":
                if isinstance(v, float) or v is None:
                    self._start_set(v, quiet=quiet)
            elif k == "delta":
                if isinstance(v, float) or v is None:
                    self._delta_set(v, quiet=quiet)
            elif k == "points":
                if isinstance(v, int) or v is None:
                    self._points_set(v, quiet=quiet)
            else:
                raise KeyError("unknown key '%s'" % k)
        return None

    @property
    def start(self) -> float | None:
        if isinstance(self.nparray, np.ndarray):
            if self.nparray.size > 0:
                return self.nparray[0]
            else:
                return None
        return self.__start

    @start.setter
    def start(self, start: float | None) -> None:
        return self._start_set(start)

    def _start_set(
        self,
        start: float | None,
        quiet: bool = nmp.QUIET
    ) -> None:
        if isinstance(self.nparray, np.ndarray):
            e = "scaling of this x-dimension is determined by a NumPy array"
            raise RuntimeError(e)
        if start is not None:
            if isinstance(start, float):
                if math.isinf(start) or math.isnan(start):
                    raise ValueError("start: %s" % start)
            elif not (isinstance(start, int) and not isinstance(start, bool)):
                e = nmu.type_error_str(start, "start", "float")
                raise TypeError(e)
        if start == self.__start:
            return None  # no change
        self.__start = start
        # h = nmh.history_change_str("start", old, start)
        # self.note = h
        # nmh.history(h, quiet=quiet)
        return None

    @property
    def delta(self) -> float | None:
        if isinstance(self.nparray, np.ndarray):
            if self.nparray.size > 1:
                return self.nparray[1] - self.nparray[0]
            else:
                return None
        return self.__delta

    @delta.setter
    def delta(self, delta: float | None) -> None:
        return self._delta_set(delta)

    def _delta_set(
        self,
        delta: float | None,
        quiet: bool = nmp.QUIET
    ) -> None:
        if isinstance(self.nparray, np.ndarray):
            e = "scaling of this x-dimension is determined by a NumPy array"
            raise RuntimeError(e)
        if delta is not None:
            if isinstance(delta, float):
                if math.isinf(delta) or math.isnan(delta):
                    raise ValueError("delta: %s" % delta)
            elif not (isinstance(delta, int) and not isinstance(delta, bool)):
                e = nmu.type_error_str(delta, "delta", "float")
                raise TypeError(e)
            if delta == 0:
                raise ValueError("delta: %s" % delta)
        if delta == self.__delta:
            return None  # no change
        self.__delta = delta
        # h = nmh.history_change_str("delta", old, delta)
        # self.note = h
        # nmh.history(h, quiet=quiet)
        return None

    @property
    def points(self) -> int | None:
        if isinstance(self.nparray, np.ndarray):
            return self.nparray.size
        if isinstance(self.__ypair, np.ndarray):
            return self.__ypair.size
        return self.__points

    @points.setter
    def points(self, points: int | None) -> None:
        return self._points_set(points)

    def _points_set(
        self,
        points: int | None,
        quiet: bool = nmp.QUIET
    ) -> None:
        if isinstance(self.nparray, np.ndarray):
            e = "scaling of this x-dimension is determined by a NumPy array"
            raise RuntimeError(e)
        if isinstance(self.ypair, np.ndarray):
            e = "scaling of this x-dimension is determined by a NumPy array"
            raise RuntimeError(e)
        if points is not None:
            if isinstance(points, float):
                if math.isnan(points) or math.isinf(points):
                    raise ValueError("points: %s" % points)
                points = int(points)
            elif isinstance(points, np.integer):
                points = int(points)
            elif not (isinstance(points, int) and not isinstance(points, bool)):
                e = nmu.type_error_str(points, "points", "integer")
                raise TypeError(e)
            if points < 0:
                raise ValueError("points: %s" % points)
        if points == self.__points:
            return None  # no change
        self.__points = points
        # h = nmh.history_change_str("points", old, points)
        # self.note = h
        # nmh.history(h, quiet=quiet)
        return None

    @property
    def ypair(self):
        return self.__ypair

    @ypair.setter
    def ypair(self, nparray) -> None:
        return self._ypair_set(nparray)

    def _ypair_set(
        self,
        nparray,
        # quiet=False
    ) -> None:
        if nparray is None:
            pass
        elif isinstance(nparray, np.ndarray):
            if isinstance(self.nparray, np.ndarray):
                if nparray.size != self.nparray.size:
                    e = ("nparray and ypair have different size: %s != %s"
                         % (self.nparray.size, nparray.size))
                    raise RuntimeError(e)
        else:
            e = nmu.type_error_str(nparray, "nparray", "np.ndarray")
            raise TypeError(e)
        self.__ypair = nparray
        return None

    def get_index(
        self,
        xvalue: float,
        clip: bool = False
    ) -> int | None:

        if not (isinstance(xvalue, (float, int)) and not isinstance(xvalue, bool)):
            e = nmu.type_error_str(xvalue, "xvalue", "float")
            raise TypeError(e)

        if isinstance(self.nparray, np.ndarray):
            if math.isinf(xvalue):
                if xvalue < 0:
                    return 0
                else:
                    return self.nparray.size - 1
            indexes = np.argwhere(self.nparray >= xvalue)
            shape = indexes.shape  # (N, 1)
            if len(shape) != 2:
                return None
            if shape[0] > 0:
                return indexes[0][0]  # first occurrence
            return None

        # Check if we have the required parameters
        if self.__start is None or self.__delta is None:
            return None

        points = self.points
        if points is None:
            return None

        if math.isinf(xvalue):
            if xvalue < 0:
                return 0
            else:
                return points - 1

        index = round((xvalue - self.__start) / self.__delta)

        if clip:
            index = max(index, 0)
            index = min(index, points - 1)
            return int(index)
        else:
            if index >= 0 and index < points:
                return int(index)
            else:
                return None

    def get_xvalue(
        self,
        index: int,
        clip: bool = False
    ) -> float | None:
        if isinstance(index, int) and not isinstance(index, bool):
            i = index
        elif isinstance(index, np.integer):
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

        points = self.points
        if points is None:
            return None

        if i >= points:
            if clip:
                i = points - 1
            else:
                e = "index out of range: %s >= %s" % (index, points)
                raise ValueError(e)

        if isinstance(self.nparray, np.ndarray):
            return self.nparray[i]

        if self.__start is None or self.__delta is None:
            return None

        return self.__start + i * self.__delta
