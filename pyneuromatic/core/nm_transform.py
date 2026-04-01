# -*- coding: utf-8 -*-
"""
NMTransform module — data transform classes for pyNeuroMatic.

Provides NMTransform base class and simple transform subclasses
(Invert, Differentiate, DoubleDifferentiate, Integrate, Log, Ln)
for use in NMChannel and NMStatWin transform pipelines.

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

import numpy as np

from pyneuromatic.core.nm_scale import NMScaleX
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_utilities as nmu


class NMTransform:
    """Base class for data transforms.

    Lightweight class (does not inherit NMObject) following the NMScaleY
    pattern. Provides serialization, deepcopy, and an apply() method
    that subclasses override.

    Transforms operate on numpy ndarrays (y-data) and optionally receive
    NMScaleX for xscale information. Simple transforms ignore xscale.
    """

    _path_suffix: str = "transform"

    def __init__(
        self,
        parent: object | None = None,
    ) -> None:
        self._parent = parent

    def __repr__(self) -> str:
        return "%s()" % self.__class__.__name__

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NMTransform):
            return self.__class__ == other.__class__
        if isinstance(other, dict):
            return self.to_dict() == other
        return NotImplemented

    def __deepcopy__(self, memo: dict) -> NMTransform:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for attr, value in self.__dict__.items():
            if attr == "_parent":
                setattr(result, attr, None)
            else:
                setattr(result, attr, copy.deepcopy(value, memo))
        return result

    def __getitem__(self, key: str) -> object:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    @property
    def path_str(self) -> str:
        if self._parent is not None and hasattr(self._parent, "path_str"):
            return self._parent.path_str + "." + self._path_suffix
        return self._path_suffix

    @property
    def type_str(self) -> str:
        """Return the transform type string used for serialization."""
        return self.__class__.__name__

    def to_dict(self) -> dict:
        """Serialize this transform to a dict.

        Returns {"type": "ClassName"}. Subclasses with parameters
        should call super().to_dict() and add their keys.
        """
        return {"type": self.type_str}

    def apply(
        self,
        ydata: np.ndarray,
        xscale: NMScaleX | None = None,
    ) -> np.ndarray:
        """Apply this transform to y-data.

        Args:
            ydata: numpy ndarray of yvalues.
            xscale: optional NMScaleX for xscale info (start, delta).

        Returns:
            numpy ndarray of transformed yvalues.

        Raises:
            NotImplementedError: if subclass does not override.
        """
        raise NotImplementedError(
            "%s.apply() not implemented" % self.__class__.__name__
        )


# =========================================================================
# Simple transforms (no parameters)
# =========================================================================


class NMTransformInvert(NMTransform):
    """Invert (negate) the data: y -> -y."""

    def apply(
        self,
        ydata: np.ndarray,
        xscale: NMScaleX | None = None,
    ) -> np.ndarray:
        if not isinstance(ydata, np.ndarray):
            raise TypeError(nmu.type_error_str(ydata, "ydata", "numpy.ndarray"))
        return -ydata


class NMTransformDifferentiate(NMTransform):
    """First derivative using numpy.gradient (preserves array length).

    Uses np.gradient() which computes central differences for interior
    points and one-sided differences at boundaries. When xscale is
    provided, divides by delta for proper dy/dx units.
    """

    def apply(
        self,
        ydata: np.ndarray,
        xscale: NMScaleX | None = None,
    ) -> np.ndarray:
        if not isinstance(ydata, np.ndarray):
            raise TypeError(nmu.type_error_str(ydata, "ydata", "numpy.ndarray"))
        if xscale is not None and isinstance(xscale, NMScaleX):
            return np.gradient(ydata, xscale.delta)
        return np.gradient(ydata)


class NMTransformDoubleDifferentiate(NMTransform):
    """Second derivative: apply np.gradient twice."""

    def apply(
        self,
        ydata: np.ndarray,
        xscale: NMScaleX | None = None,
    ) -> np.ndarray:
        if not isinstance(ydata, np.ndarray):
            raise TypeError(nmu.type_error_str(ydata, "ydata", "numpy.ndarray"))
        if xscale is not None and isinstance(xscale, NMScaleX):
            d1 = np.gradient(ydata, xscale.delta)
            return np.gradient(d1, xscale.delta)
        d1 = np.gradient(ydata)
        return np.gradient(d1)


class NMTransformIntegrate(NMTransform):
    """Cumulative integration using numpy.cumsum.

    When xscale is provided, multiplies by delta for proper units.
    """

    def apply(
        self,
        ydata: np.ndarray,
        xscale: NMScaleX | None = None,
    ) -> np.ndarray:
        if not isinstance(ydata, np.ndarray):
            raise TypeError(nmu.type_error_str(ydata, "ydata", "numpy.ndarray"))
        result = np.cumsum(ydata)
        if xscale is not None and isinstance(xscale, NMScaleX):
            result = result * xscale.delta
        return result


class NMTransformLog(NMTransform):
    """Base-10 logarithm: y -> log10(y).

    Values <= 0 produce -inf (zero) or NaN (negative).
    """

    def apply(
        self,
        ydata: np.ndarray,
        xscale: NMScaleX | None = None,
    ) -> np.ndarray:
        if not isinstance(ydata, np.ndarray):
            raise TypeError(nmu.type_error_str(ydata, "ydata", "numpy.ndarray"))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(ydata)


class NMTransformLn(NMTransform):
    """Natural logarithm: y -> ln(y).

    Values <= 0 produce -inf (zero) or NaN (negative).
    """

    def apply(
        self,
        ydata: np.ndarray,
        xscale: NMScaleX | None = None,
    ) -> np.ndarray:
        if not isinstance(ydata, np.ndarray):
            raise TypeError(nmu.type_error_str(ydata, "ydata", "numpy.ndarray"))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(ydata)


# =========================================================================
# Smooth transform
# =========================================================================

_VALID_SMOOTH_METHODS: frozenset[str] = frozenset({"boxcar", "binomial", "savgol"})


class NMTransformSmooth(NMTransform):
    """Smooth transform: boxcar, binomial, or Savitzky-Golay (polynomial).

    Applies the chosen smooth method to y-data via the shared pure functions
    in :mod:`pyneuromatic.core.nm_math`.

    Args:
        method: ``"boxcar"``, ``"binomial"``, or ``"savgol"``. Default ``"boxcar"``.
        window: Kernel width in points (odd int >= 3). Used by boxcar and savgol.
            Default 5.
        passes: Number of times to apply the kernel (int >= 1). Default 1.
            Used by boxcar and binomial; ignored by savgol.
        polyorder: Polynomial order for savgol (int >= 1, < window). Default 2.
    """

    def __init__(
        self,
        parent: object | None = None,
        method: str = "boxcar",
        window: int = 5,
        passes: int = 1,
        polyorder: int = 2,
    ) -> None:
        super().__init__(parent=parent)
        self.method = method
        self.window = window
        self.passes = passes
        self.polyorder = polyorder

    def __repr__(self) -> str:
        return (
            "%s(method=%r, window=%r, passes=%r, polyorder=%r)"
            % (self.__class__.__name__, self._method, self._window,
               self._passes, self._polyorder)
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NMTransformSmooth):
            return (
                self._method == other._method
                and self._window == other._window
                and self._passes == other._passes
                and self._polyorder == other._polyorder
            )
        if isinstance(other, dict):
            return self.to_dict() == other
        return NotImplemented

    # ------------------------------------------------------------------
    # Properties

    @property
    def method(self) -> str:
        """Smooth method: ``'boxcar'``, ``'binomial'``, or ``'savgol'``."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "method", "string"))
        if value not in _VALID_SMOOTH_METHODS:
            raise ValueError(
                "method must be one of %s, got %r"
                % (sorted(_VALID_SMOOTH_METHODS), value)
            )
        self._method = value

    @property
    def window(self) -> int:
        """Kernel width in points (odd int >= 3). Used by boxcar and savgol."""
        return self._window

    @window.setter
    def window(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "window", "int"))
        if value < 3:
            raise ValueError("window must be >= 3, got %d" % value)
        if value % 2 == 0:
            raise ValueError("window must be odd, got %d" % value)
        self._window = value

    @property
    def passes(self) -> int:
        """Number of times to apply the kernel (int >= 1). Used by boxcar and binomial."""
        return self._passes

    @passes.setter
    def passes(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "passes", "int"))
        if value < 1:
            raise ValueError("passes must be >= 1, got %d" % value)
        self._passes = value

    @property
    def polyorder(self) -> int:
        """Polynomial order for savgol (int >= 1, < window)."""
        return self._polyorder

    @polyorder.setter
    def polyorder(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "polyorder", "int"))
        if value < 1:
            raise ValueError("polyorder must be >= 1, got %d" % value)
        self._polyorder = value

    # ------------------------------------------------------------------
    # Core

    def apply(
        self,
        ydata: np.ndarray,
        xscale: NMScaleX | None = None,
    ) -> np.ndarray:
        """Apply the smooth to *ydata* and return the result."""
        import pyneuromatic.core.nm_math as nm_math
        if self._method == "savgol" and self._passes > 1:
            nmh.history(
                "passes=%d ignored for savgol; savgol is applied once" % self._passes,
                title="ALERT",
                red=True,
            )
        if self._method == "boxcar":
            return nm_math.smooth_boxcar(ydata, self._window, self._passes)
        if self._method == "binomial":
            return nm_math.smooth_binomial(ydata, self._passes)
        # savgol
        return nm_math.smooth_savgol(ydata, self._window, self._polyorder)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "method": self._method,
            "window": self._window,
            "passes": self._passes,
            "polyorder": self._polyorder,
        })
        return d


# =========================================================================
# Registry and helper functions
# =========================================================================


_TRANSFORM_REGISTRY: dict[str, type[NMTransform]] = {
    "NMTransformInvert": NMTransformInvert,
    "NMTransformDifferentiate": NMTransformDifferentiate,
    "NMTransformDoubleDifferentiate": NMTransformDoubleDifferentiate,
    "NMTransformIntegrate": NMTransformIntegrate,
    "NMTransformLog": NMTransformLog,
    "NMTransformLn": NMTransformLn,
    "NMTransformSmooth": NMTransformSmooth,
}


def _transform_from_dict(
    d: dict,
    parent: object | None = None,
) -> NMTransform:
    """Reconstruct an NMTransform from a dict.

    Args:
        d: Dict with at least a "type" key naming the transform class.
        parent: Optional parent reference.

    Returns:
        An NMTransform instance.

    Raises:
        TypeError: if d is not a dict.
        KeyError: if "type" key is missing.
        ValueError: if type string is not in the registry.
    """
    if not isinstance(d, dict):
        raise TypeError(nmu.type_error_str(d, "d", "dict"))
    if "type" not in d:
        raise KeyError("missing key 'type' in transform dict")
    type_str = d["type"]
    if type_str not in _TRANSFORM_REGISTRY:
        raise ValueError("unknown transform type: '%s'" % type_str)
    cls = _TRANSFORM_REGISTRY[type_str]
    kwargs = {k: v for k, v in d.items() if k != "type"}
    return cls(parent=parent, **kwargs)


def _transforms_from_list(
    transform_list: list[dict] | None,
    parent: object | None = None,
) -> list[NMTransform] | None:
    """Reconstruct a list of NMTransform objects from a list of dicts.

    Args:
        transform_list: List of transform dicts, or None.
        parent: Optional parent reference.

    Returns:
        List of NMTransform instances, or None.
    """
    if transform_list is None:
        return None
    if not isinstance(transform_list, list):
        raise TypeError(
            nmu.type_error_str(transform_list, "transform_list", "list")
        )
    return [_transform_from_dict(d, parent=parent) for d in transform_list]


def apply_transforms(
    ydata: np.ndarray,
    transforms: list[NMTransform] | None,
    xscale: NMScaleX | None = None,
) -> np.ndarray:
    """Apply an ordered list of transforms to y-data.

    Args:
        ydata: numpy ndarray of yvalues.
        transforms: ordered list of NMTransform objects, or None.
        xscale: optional NMScaleX for xscale info.

    Returns:
        Transformed numpy ndarray (copy; original is never modified).
    """
    if not isinstance(ydata, np.ndarray):
        raise TypeError(nmu.type_error_str(ydata, "ydata", "numpy.ndarray"))
    if transforms is None or len(transforms) == 0:
        return ydata
    result = ydata.copy()
    for t in transforms:
        if not isinstance(t, NMTransform):
            raise TypeError(nmu.type_error_str(t, "transform", "NMTransform"))
        result = t.apply(result, xscale=xscale)
    return result
