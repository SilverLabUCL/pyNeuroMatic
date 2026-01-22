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
import math
import numpy as np

from pyneuromatic.core.nm_object import NMObject
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu


class NMDimension(NMObject):
    """
    NM Scale class

    Contains data y-scale parameters
    """

    def __init__(
        self,
        parent: object = None,
        name: str = "NMDimension0",
        nparray=None,  # 1D TODO: typing
        scale: dict | None = None,  # "label" and "units"
        copy: NMDimension | None = None,  # see copy()
    ) -> None:
        super().__init__(parent=parent, name=name, copy=copy)

        if copy is None:
            pass  # ok
        elif isinstance(copy, NMDimension):
            if isinstance(copy.nparray, np.ndarray):
                nparray = copy.nparray.copy()
            scale = copy.scale
        else:
            e = nmu.typeerror(copy, "copy", NMDimension)
            raise TypeError(e)

        if nparray is None:
            pass
        elif isinstance(nparray, np.ndarray):
            if nparray.ndim != 1:
                e = ("NumPy array should have 1 dimension, not %s" %
                     nparray.ndim)
                raise ValueError(e)
        else:
            e = nmu.typeerror(nparray, "nparray", "np.ndarray")
            raise TypeError(e)

        self.__nparray = nparray
        self.__label = None
        self.__units = None

        if scale is None:
            pass  # ok
        elif isinstance(scale, dict):
            if not copy:
                self._scale_set(scale, quiet=True)
        else:
            e = nmu.typeerror(scale, "scale", "dictionary")
            raise TypeError(e)

        return None

    # override
    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if self.__label.lower() != other.label.lower():
            return False
        if self.__units.lower() != other.units.lower():
            return False
        if not NMDimension.__eq_arrays(self.__nparray, other.nparray):
            return False
        return True

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

    # override, no super
    def copy(self) -> NMDimension:
        return NMDimension(copy=self)

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
            e = nmu.typeerror(scale, "scale", "dictionary")
            raise TypeError(e)
        for k, v in scale.items():
            k = k.lower()
            if k == "label":
                self._label_set(v, quiet=quiet)
            elif k == "units":
                self._units_set(v, quiet=quiet)
            else:
                raise KeyError("unknown key '%s'" % k)
        return None

    @property
    def label(self) -> str:
        return self.__label

    @label.setter
    def label(self, label: str) -> None:
        return self._label_set(label)

    def _label_set(self, label: str, quiet: bool = nmp.QUIET) -> None:
        if label is None:
            pass
        elif not isinstance(label, str):
            e = nmu.typeerror(label, "label", "string")
            raise TypeError(e)
        old = self.__label
        if label == old:
            return None
        self.__label = label
        # h = nmu.history_change("label", old, label)
        # self.note = h
        # self._history(h, quiet=quiet)
        return None

    @property
    def units(self) -> str:
        return self.__units

    @units.setter
    def units(self, units: str) -> None:
        return self._units_set(units)

    def _units_set(self, units: str, quiet: bool = nmp.QUIET) -> None:
        if units is None:
            pass
        elif not isinstance(units, str):
            e = nmu.typeerror(units, "units", "string")
            raise TypeError(e)
        old = self.__units
        if units == old:
            return None
        self.__units = units
        # h = nmu.history_change("units", old, units)
        # self.note = h
        # self._history(h, quiet=quiet)
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
        # quiet=nmp.QUIET
    ) -> None:
        if nparray is None:
            pass  # ok
        elif not isinstance(nparray, np.ndarray):
            e = nmu.typeerror(nparray, "nparray", "np.ndarray")
            raise TypeError(e)
        self.__nparray = nparray
        return None


class NMDimensionX(NMDimension):
    """
    NM X-Scale class

    Contains data x-scale parameters
    """

    def __init__(
        self,
        parent: object = None,
        name: str = "NMDimensionX0",
        nparray=None,  # TODO: typing
        ypair=None,  # TODO: typing
        scale: dict | None = None,
        # "label", "units", "start", "delta", "points"
        copy: NMDimensionX = None,  # see copy()
    ) -> None:

        # declare parameters before calling super (so scale_set() is OK)
        self.__start = None
        self.__delta = None
        self.__points = None
        self.__ypair = None

        super().__init__(
                parent=parent,
                name=name,
                nparray=nparray,
                scale=scale,
                copy=copy
        )

        self._ypair_set(ypair)

        return None

    # override
    def __eq__(self, other: object) -> bool:
        if not super().__eq__(other):
            return False
        if self.start != other.start:
            return False
        if self.delta != other.delta:
            return False
        if self.points != other.points:
            return False
        return True

    # override, no super
    def copy(self) -> NMDimensionX:
        return NMDimensionX(copy=self)

    # override, no super
    def _scale_set(
        self,
        scale: dict[str, object],
        quiet: bool = nmp.QUIET
    ) -> None:
        super()._scale_set(scale)
        for k, v in scale.items():
            k = k.lower()
            if k == "start":
                self._start_set(v, quiet=quiet)
            elif k == "delta":
                self._delta_set(v, quiet=quiet)
            elif k == "points":
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
        if start is None:
            pass
        elif isinstance(start, float):
            if math.isinf(start) or math.isnan(start):
                raise ValueError("start: %s" % start)
        elif isinstance(start, int) and not isinstance(start, bool):
            pass
        else:
            e = nmu.typeerror(start, "start", "float")
            raise TypeError(e)
        if start == self.__start:
            return None
        # old = self.__start
        self.__start = start  # mangled
        # h = nmu.history_change("start", old, start)
        # self.note = h
        # self._history(h, quiet=quiet)
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
        if delta is None:
            pass
        elif isinstance(delta, float):
            if math.isinf(delta) or math.isnan(delta):
                raise ValueError("delta: %s" % delta)
        elif isinstance(delta, int) and not isinstance(delta, bool):
            pass
        else:
            e = nmu.typeerror(delta, "delta", "float")
            raise TypeError(e)
        if delta and delta == 0:
            raise ValueError("delta: %s" % delta)
        if delta == self.__delta:
            return None
        # old = self.__delta
        self.__delta = delta
        # h = nmu.history_change("delta", old, delta)
        # self.note = h
        # self._history(h, quiet=quiet)
        return None

    @property
    def points(self) -> int:
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
        if points is None:
            pass
        elif isinstance(points, float) or isinstance(points, np.float):
            if math.isnan(points) or math.isinf(points):
                raise ValueError("points: %s" % points)
            points = int(points)
        elif isinstance(points, int) and not isinstance(points, bool):
            pass
        elif isinstance(points, np.integer):
            points = int(points)
        else:
            e = nmu.typeerror(points, "points", "integer")
            raise TypeError(e)
        if points and points < 0:
            raise ValueError("points: %s" % points)
        if points == self.__points:
            return None
        # old = self.__points
        self.__points = points
        # h = nmu.history_change("delta", old, delta)
        # self.note = h
        # self._history(h, quiet=quiet)
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
        # quiet=nmp.QUIET
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
            e = nmu.typeerror(nparray, "nparray", "np.ndarray")
            raise TypeError(e)
        self.__ypair = nparray
        return None

    def get_index(self, xvalue: float, clip: bool = False) -> int | None:

        if isinstance(xvalue, float):
            pass
        elif isinstance(xvalue, int) and not isinstance(xvalue, bool):
            pass
        else:
            e = nmu.typeerror(xvalue, "xvalue", "float")
            raise TypeError(e)

        if math.isinf(xvalue):  # clip = True
            if xvalue < 0:
                return 0
            else:
                return self.points - 1

        if isinstance(self.nparray, np.ndarray):
            indexes = np.argwhere(self.nparray >= xvalue)
            shape = indexes.shape  # (N, 1)
            if len(shape) != 2:
                return None
            if shape[0] > 0:
                return indexes[0][0]  # first occurrence
            return None

        index = round((xvalue - self.__start) / self.__delta)
        points = self.points

        if clip:
            index = max(index, 0)
            index = min(index, points - 1)
            return int(index)
        else:
            if index >= 0 and index < points:
                return index
            else:
                return None

    def get_xvalue(self, index: int, clip: bool = False) -> float:

        if isinstance(index, int) and not isinstance(index, bool):
            i = index
        elif isinstance(index, np.integer):
            i = int(index)
        elif isinstance(index, float):
            i = int(index)
        else:
            e = nmu.typeerror(index, "index", "integer")
            raise TypeError(e)

        if i < 0:
            if clip:
                i = 0
            else:
                raise ValueError("negative index: %s" % index)

        points = self.points
        if i >= points:
            if clip:
                i = points - 1
            else:
                e = "index out or range: %s >= %s" % (index, points)
                raise ValueError(e)

        if isinstance(self.nparray, np.ndarray):
            return self.nparray[i]

        return self.__start + i * self.__delta
