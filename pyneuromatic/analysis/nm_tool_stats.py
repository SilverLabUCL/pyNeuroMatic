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
from typing import Any
import numpy as np

from pyneuromatic.analysis.nm_tool_folder import NMToolFolder
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_preferences as nmp
from pyneuromatic.analysis.nm_tool import NMTool
from pyneuromatic.core.nm_transform import (
    NMTransform,
    apply_transforms,
    _transform_from_dict,
)
import pyneuromatic.core.nm_utilities as nmu

FUNC_NAMES = (
    "max",  # np.argmax
    "min",  # np.argmin
    "mean@max",
    "mean@min",
    "median",  # np.median
    "mean",  # "avg" # np.mean
    "mean+var",
    "mean+std",
    "mean+sem",
    "var",  # np.var
    "std",  # "sdev" # np.std
    "sem",
    "rms",
    "sum",  # np.sum
    "pathlength",
    "area",
    "slope",
    # TODO "onset",  # Igor version requires sigmoid curvefit
    "level",
    "level+",
    "level-",
    "value@x0",
    "value@x1",
    "count",  # "numpnts"
    "count_nans",
    "count_infs",
    # positive peaks
    "risetime+",
    "falltime+",
    "risetimeslope+",
    "falltimeslope+",
    "fwhm+",
    # negative peaks
    "risetime-",
    "falltime-",
    "risetimeslope-",
    "falltimeslope-",
    "fwhm-"
)

FUNC_NAMES_BSLN = (
    "median",
    "mean",
    "mean+var",
    "mean+std",
    "mean+sem",
    # "var",
    # "std",
    # "sem",
    # "rms",
    # "sum",
    # "pathlength",
    # "area",
    # "slope",
)


def badvalue(n: float | None) -> bool:
    return n is None or math.isnan(n) or math.isinf(n)


# =========================================================================
# Stats function classes
# =========================================================================

FUNC_NAMES_BASIC = (
    "median", "mean", "mean+var", "mean+std", "mean+sem",
    "var", "std", "sem", "rms", "sum", "pathlength", "area", "slope",
    "value@x0", "value@x1", "count", "count_nans", "count_infs",
)

FUNC_NAMES_MAXMIN = ("max", "min", "mean@max", "mean@min")

FUNC_NAMES_LEVEL = ("level", "level+", "level-")

FUNC_NAMES_RISETIME = (
    "risetime+", "risetime-", "risetimeslope+", "risetimeslope-",
)

FUNC_NAMES_FALLTIME = (
    "falltime+", "falltime-", "falltimeslope+", "falltimeslope-",
)

FUNC_NAMES_FWHM = ("fwhm+", "fwhm-")


class NMStatsFunc:
    """Base class for stats function types.

    Lightweight class (following NMTransform pattern) that represents
    a statistics function with its parameters and compute pipeline.
    """

    _VALID_KEYS: set[str] = set()

    def __init__(self, name: str, parent: object | None = None) -> None:
        self._parent = parent
        self._name = name

    def __repr__(self) -> str:
        d = self.to_dict()
        params = ", ".join("%s=%r" % (k, v) for k, v in d.items())
        return "%s(%s)" % (self.__class__.__name__, params)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NMStatsFunc):
            return self.to_dict() == other.to_dict()
        if isinstance(other, dict):
            return self.to_dict() == other
        return NotImplemented

    def __deepcopy__(self, memo: dict) -> NMStatsFunc:
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
    def name(self) -> str:
        return self._name

    def to_dict(self) -> dict:
        return {"name": self._name}

    @property
    def needs_baseline(self) -> bool:
        return False

    def validate_baseline(self, bsln_func_name: str | None) -> None:
        pass

    def _add_ds(self, r: dict, bsln_result: dict) -> float:
        """Compute Δs (stat minus baseline) and add to result dict."""
        ds = math.nan
        if bsln_result:
            if "s" in bsln_result:
                bs = bsln_result["s"]
                if "s" in r:
                    rs = r["s"]
                elif isinstance(r.get("func"), dict) and "ylevel" in r["func"]:
                    rs = r["func"]["ylevel"]
                else:
                    rs = math.nan
                if not badvalue(bs) and not badvalue(rs):
                    ds = rs - bs
            r["Δs"] = ds
        return ds

    def compute(self, data, x0, x1, xclip, ignore_nans, run_stat,
                bsln_result):
        raise NotImplementedError(
            "%s.compute() not implemented" % self.__class__.__name__
        )


class NMStatsFuncBasic(NMStatsFunc):
    """Stats functions with no parameters (mean, median, var, etc.)."""

    def __init__(self, name: str, parent: object | None = None) -> None:
        if name.lower() not in FUNC_NAMES_BASIC:
            raise ValueError("func name: '%s'" % name)
        super().__init__(name.lower(), parent=parent)

    def compute(self, data, x0, x1, xclip, ignore_nans, run_stat,
                bsln_result):
        r = run_stat(data, {"name": self._name}, "main",
                     x0, x1, xclip, ignore_nans)
        self._add_ds(r, bsln_result)


class NMStatsFuncMaxMin(NMStatsFunc):
    """max, min, mean@max, mean@min functions."""

    _VALID_KEYS = {"n_avg"}

    def __init__(
        self, name: str, n_avg: int | None = None,
        parent: object | None = None
    ) -> None:
        if name.lower() not in FUNC_NAMES_MAXMIN:
            raise ValueError("func name: '%s'" % name)
        f = name.lower()
        if f in ("mean@max", "mean@min"):
            if n_avg is None:
                raise KeyError("missing func key 'n_avg'")
        if n_avg is not None:
            if isinstance(n_avg, bool):
                raise TypeError("n_avg: '%s'" % n_avg)
            n_avg = int(n_avg)  # raises TypeError/ValueError/OverflowError
            if n_avg < 0:
                raise ValueError("n_avg: '%s'" % n_avg)
            # Upgrade max→mean@max, min→mean@min when n_avg is provided
            if f == "max":
                f = "mean@max"
            elif f == "min":
                f = "mean@min"
        super().__init__(f, parent=parent)
        self._n_avg = n_avg

    def to_dict(self) -> dict:
        d = {"name": self._name}
        if self._n_avg is not None:
            d["n_avg"] = self._n_avg
        return d

    def compute(self, data, x0, x1, xclip, ignore_nans, run_stat,
                bsln_result):
        func: dict[str, Any] = {"name": self._name}
        if self._n_avg is not None:
            func["n_avg"] = self._n_avg
        if self._name in ("mean@max", "mean@min"):
            n_avg = self._n_avg if self._n_avg is not None else 0
            if n_avg <= 1:
                func["warning"] = "not enough data points to compute a mean"
        r = run_stat(data, func, "main", x0, x1, xclip, ignore_nans)
        self._add_ds(r, bsln_result)


class NMStatsFuncLevel(NMStatsFunc):
    """Level with explicit ylevel value."""

    _VALID_KEYS = {"ylevel"}

    def __init__(
        self, name: str, ylevel: float | None = None,
        parent: object | None = None
    ) -> None:
        if name.lower() not in FUNC_NAMES_LEVEL:
            raise ValueError("func name: '%s'" % name)
        if ylevel is None:
            raise KeyError("missing func key 'ylevel'")
        if isinstance(ylevel, bool):
            raise TypeError("ylevel: '%s'" % ylevel)
        ylevel = float(ylevel)
        if math.isnan(ylevel) or math.isinf(ylevel):
            raise ValueError("ylevel: '%s'" % ylevel)
        super().__init__(name.lower(), parent=parent)
        self._ylevel = ylevel

    def to_dict(self) -> dict:
        return {"name": self._name, "ylevel": self._ylevel}

    def compute(self, data, x0, x1, xclip, ignore_nans, run_stat,
                bsln_result):
        func: dict[str, Any] = {"name": self._name, "ylevel": self._ylevel}
        r = run_stat(data, func, "main", x0, x1, xclip, ignore_nans)
        self._add_ds(r, bsln_result)


class NMStatsFuncLevelNstd(NMStatsFunc):
    """Level computed from baseline mean +/- n_std * std."""

    _VALID_KEYS = {"n_std"}

    def __init__(
        self, name: str, n_std: float | None = None,
        parent: object | None = None
    ) -> None:
        if name.lower() not in FUNC_NAMES_LEVEL:
            raise ValueError("func name: '%s'" % name)
        if n_std is None:
            raise KeyError("missing func key 'n_std'")
        if isinstance(n_std, bool):
            raise TypeError("n_std: '%s'" % n_std)
        n_std = float(n_std)
        if math.isnan(n_std) or math.isinf(n_std) or n_std == 0:
            raise ValueError("n_std: '%s'" % n_std)
        super().__init__(name.lower(), parent=parent)
        self._n_std = n_std

    def to_dict(self) -> dict:
        return {"name": self._name, "n_std": self._n_std}

    @property
    def needs_baseline(self) -> bool:
        return True

    def validate_baseline(self, bsln_func_name: str | None) -> None:
        if bsln_func_name is None or bsln_func_name.lower() != "mean+std":
            raise RuntimeError(
                "level n_std requires baseline 'mean+std' computation"
            )

    def compute(self, data, x0, x1, xclip, ignore_nans, run_stat,
                bsln_result):
        ylevel = math.nan
        if "s" in bsln_result and "std" in bsln_result:
            s = bsln_result["s"]
            std = bsln_result["std"]
            if not badvalue(s) and not badvalue(std):
                ylevel = s + self._n_std * std
        func: dict[str, Any] = {"name": self._name, "ylevel": ylevel}
        r = run_stat(data, func, "main", x0, x1, xclip, ignore_nans)
        self._add_ds(r, bsln_result)


class NMStatsFuncRiseTime(NMStatsFunc):
    """Rise time functions (risetime+/-, risetimeslope+/-)."""

    _VALID_KEYS = {"p0", "p1"}

    def __init__(
        self, name: str, p0: float | None = None,
        p1: float | None = None, parent: object | None = None
    ) -> None:
        if name.lower() not in FUNC_NAMES_RISETIME:
            raise ValueError("func name: '%s'" % name)
        f = name.lower()
        if p0 is None:
            raise KeyError("missing func key 'p0'")
        if isinstance(p0, bool):
            raise TypeError("p0: '%s'" % p0)
        p0 = float(p0)
        if math.isnan(p0) or math.isinf(p0):
            raise ValueError("p0: '%s'" % p0)
        if not (p0 > 0 and p0 < 100):
            raise ValueError("bad percent p0: %s" % p0)
        if p1 is None:
            raise KeyError("missing func key 'p1'")
        if isinstance(p1, bool):
            raise TypeError("p1: '%s'" % p1)
        p1 = float(p1)
        if math.isnan(p1) or math.isinf(p1):
            raise ValueError("p1: '%s'" % p1)
        if not (p1 > 0 and p1 < 100):
            raise ValueError("bad percent p1: %s" % p1)
        if p0 >= p1:
            raise ValueError(
                "for risetime, need p0 < p1 but got %s >= %s" % (p0, p1))
        super().__init__(f, parent=parent)
        self._p0 = p0
        self._p1 = p1

    def to_dict(self) -> dict:
        return {"name": self._name, "p0": self._p0, "p1": self._p1}

    @property
    def needs_baseline(self) -> bool:
        return True

    def validate_baseline(self, bsln_func_name: str | None) -> None:
        if bsln_func_name is None:
            raise RuntimeError(
                "peak func '%s' requires baseline 'mean' computation"
                % self._name)
        bf = bsln_func_name.lower()
        if "mean" not in bf and bf != "median":
            raise RuntimeError(
                "peak func '%s' requires baseline 'mean' computation"
                % self._name)

    def compute(self, data, x0, x1, xclip, ignore_nans, run_stat,
                bsln_result):
        f = self._name
        slope = "slope" in f

        # Peak stat
        if "+" in f:
            peak_func: dict[str, Any] = {"name": "max"}
            flevel: dict[str, Any] = {"name": "level+"}
        else:
            peak_func = {"name": "min"}
            flevel = {"name": "level-"}

        r = run_stat(data, peak_func, f, x0, x1, xclip, ignore_nans)
        ds = self._add_ds(r, bsln_result)

        if badvalue(ds):
            r["error"] = "unable to compute peak height Δs"
            return

        peak_x = r["x"]

        flevel["ylevel"] = 0.01 * self._p0 * ds
        r0 = run_stat(data, flevel, f, x0, peak_x, xclip, ignore_nans,
                      p0=self._p0)
        r0_error = "x" not in r0 or badvalue(r0["x"])

        flevel["ylevel"] = 0.01 * self._p1 * ds
        r1 = run_stat(data, flevel, f, x0, peak_x, xclip, ignore_nans,
                      p1=self._p1)
        r1_error = "x" not in r1 or badvalue(r1["x"])

        if r1_error:
            r1["error"] = "unable to locate p1 level"
            r1["dx"] = math.nan
            return

        if r0_error:
            r1["dx"] = math.nan
            return
        r1["dx"] = r1["x"] - r0["x"]

        if slope:
            run_stat(data, {"name": "slope"}, f, r0["x"], r1["x"],
                     xclip, ignore_nans)


class NMStatsFuncFallTime(NMStatsFunc):
    """Fall time functions (falltime+/-, falltimeslope+/-)."""

    _VALID_KEYS = {"p0", "p1"}

    def __init__(
        self, name: str, p0: float | None = None,
        p1: float | None = None, parent: object | None = None
    ) -> None:
        if name.lower() not in FUNC_NAMES_FALLTIME:
            raise ValueError("func name: '%s'" % name)
        f = name.lower()
        if p0 is None:
            raise KeyError("missing func key 'p0'")
        if isinstance(p0, bool):
            raise TypeError("p0: '%s'" % p0)
        p0 = float(p0)
        if math.isnan(p0) or math.isinf(p0):
            raise ValueError("p0: '%s'" % p0)
        if not (p0 > 0 and p0 < 100):
            raise ValueError("bad percent p0: %s" % p0)
        if p1 is not None:
            if isinstance(p1, bool):
                raise TypeError("p1: '%s'" % p1)
            p1 = float(p1)
            if math.isnan(p1) or math.isinf(p1):
                raise ValueError("p1: '%s'" % p1)
            if not (p1 > 0 and p1 < 100):
                raise ValueError("bad percent p1: %s" % p1)
            if p0 <= p1:
                raise ValueError(
                    "for falltime, need p0 > p1 but got %s <= %s" % (p0, p1))
        super().__init__(f, parent=parent)
        self._p0 = p0
        self._p1 = p1

    def to_dict(self) -> dict:
        return {"name": self._name, "p0": self._p0, "p1": self._p1}

    @property
    def needs_baseline(self) -> bool:
        return True

    def validate_baseline(self, bsln_func_name: str | None) -> None:
        if bsln_func_name is None:
            raise RuntimeError(
                "peak func '%s' requires baseline 'mean' computation"
                % self._name)
        bf = bsln_func_name.lower()
        if "mean" not in bf and bf != "median":
            raise RuntimeError(
                "peak func '%s' requires baseline 'mean' computation"
                % self._name)

    def compute(self, data, x0, x1, xclip, ignore_nans, run_stat,
                bsln_result):
        f = self._name
        slope = "slope" in f

        # Peak stat
        if "+" in f:
            peak_func: dict[str, Any] = {"name": "max"}
            flevel: dict[str, Any] = {"name": "level-"}  # opposite sign
        else:
            peak_func = {"name": "min"}
            flevel = {"name": "level+"}  # opposite sign

        r = run_stat(data, peak_func, f, x0, x1, xclip, ignore_nans)
        ds = self._add_ds(r, bsln_result)

        if badvalue(ds):
            r["error"] = "unable to compute peak height Δs"
            return

        peak_x = r["x"]

        flevel["ylevel"] = 0.01 * self._p0 * ds
        r0 = run_stat(data, flevel, f, peak_x, x1, xclip, ignore_nans,
                      p0=self._p0)
        r0_error = "x" not in r0 or badvalue(r0["x"])
        if r0_error:
            r0["error"] = "unable to locate p0 level"

        if self._p1 is None:
            if r0_error:
                r0["dx"] = math.nan
                return
            r0["dx"] = r0["x"] - peak_x
        else:
            flevel["ylevel"] = 0.01 * self._p1 * ds
            r1 = run_stat(data, flevel, f, peak_x, x1, xclip, ignore_nans,
                          p1=self._p1)
            r1_error = "x" not in r1 or badvalue(r1["x"])

            if r1_error:
                r1["error"] = "unable to locate p1 level"
                r1["dx"] = math.nan
                return
            if r0_error:
                r1["dx"] = math.nan
                return
            r1["dx"] = r1["x"] - r0["x"]

        if slope:
            if self._p1 is not None:
                run_stat(data, {"name": "slope"}, f, r0["x"], r1["x"],
                         xclip, ignore_nans)
            else:
                run_stat(data, {"name": "slope"}, f, peak_x, r0["x"],
                         xclip, ignore_nans)


class NMStatsFuncFWHM(NMStatsFunc):
    """Full width at half maximum functions."""

    _VALID_KEYS = {"p0", "p1"}

    def __init__(
        self, name: str, p0: float | None = None,
        p1: float | None = None, parent: object | None = None
    ) -> None:
        if name.lower() not in FUNC_NAMES_FWHM:
            raise ValueError("func_name: '%s'" % name)
        f = name.lower()
        if p0 is None:
            p0 = 50
        elif isinstance(p0, bool):
            raise TypeError("p0: '%s'" % p0)
        if p1 is None:
            p1 = 50
        elif isinstance(p1, bool):
            raise TypeError("p1: '%s'" % p1)
        p0 = float(p0)
        p1 = float(p1)
        if not (p0 > 0 and p0 < 100):
            raise ValueError("p0: %s" % p0)
        if not (p1 > 0 and p1 < 100):
            raise ValueError("p1: %s" % p1)
        super().__init__(f, parent=parent)
        self._p0 = p0
        self._p1 = p1

    def to_dict(self) -> dict:
        return {"name": self._name, "p0": self._p0, "p1": self._p1}

    @property
    def needs_baseline(self) -> bool:
        return True

    def validate_baseline(self, bsln_func_name: str | None) -> None:
        if bsln_func_name is None:
            raise RuntimeError(
                "peak func '%s' requires baseline 'mean' computation"
                % self._name)
        bf = bsln_func_name.lower()
        if "mean" not in bf and bf != "median":
            raise RuntimeError(
                "peak func '%s' requires baseline 'mean' computation"
                % self._name)

    def compute(self, data, x0, x1, xclip, ignore_nans, run_stat,
                bsln_result):
        f = self._name

        # Peak stat
        if "+" in f:
            peak_func: dict[str, Any] = {"name": "max"}
        else:
            peak_func = {"name": "min"}

        r = run_stat(data, peak_func, f, x0, x1, xclip, ignore_nans)

        # Compute Δs from baseline
        ds = self._add_ds(r, bsln_result)

        if badvalue(ds):
            r["error"] = "unable to compute peak height Δs"
            return

        if "+" in f:
            flevel1: dict[str, Any] = {"name": "level+"}
            flevel2: dict[str, Any] = {"name": "level-"}  # opposite sign
        else:
            flevel1 = {"name": "level-"}
            flevel2 = {"name": "level+"}  # opposite sign

        w = None
        if self._p0 != 50 or self._p1 != 50:
            w = "unusual fwhm %% values: %s-%s" % (self._p0, self._p1)

        flevel1["ylevel"] = 0.01 * self._p0 * ds
        peak_x = r["x"]
        extra0: dict[str, Any] = {"p0": self._p0}
        if w:
            extra0["warning"] = w
        r0 = run_stat(data, flevel1, f, x0, peak_x, xclip, ignore_nans,
                      **extra0)
        r0_error = "x" not in r0 or badvalue(r0["x"])
        if r0_error:
            r0["error"] = "unable to locate p0 level"

        flevel2["ylevel"] = 0.01 * self._p1 * ds
        extra1: dict[str, Any] = {"p1": self._p1}
        if w:
            extra1["warning"] = w
        r1 = run_stat(data, flevel2, f, peak_x, x1, xclip, ignore_nans,
                      **extra1)
        r1_error = "x" not in r1 or badvalue(r1["x"])

        if r1_error:
            r1["error"] = "unable to locate p1 level"
            r1["dx"] = math.nan
            return

        if r0_error:
            r1["dx"] = math.nan
        else:
            r1["dx"] = r1["x"] - r0["x"]


# Registry mapping func names to their class
_STATS_FUNC_REGISTRY: dict[str, type[NMStatsFunc]] = {}
for _name in FUNC_NAMES_BASIC:
    _STATS_FUNC_REGISTRY[_name] = NMStatsFuncBasic
for _name in FUNC_NAMES_MAXMIN:
    _STATS_FUNC_REGISTRY[_name] = NMStatsFuncMaxMin
for _name in FUNC_NAMES_LEVEL:
    _STATS_FUNC_REGISTRY[_name] = NMStatsFuncLevel
for _name in FUNC_NAMES_RISETIME:
    _STATS_FUNC_REGISTRY[_name] = NMStatsFuncRiseTime
for _name in FUNC_NAMES_FALLTIME:
    _STATS_FUNC_REGISTRY[_name] = NMStatsFuncFallTime
for _name in FUNC_NAMES_FWHM:
    _STATS_FUNC_REGISTRY[_name] = NMStatsFuncFWHM


def _stats_func_from_dict(
    d: dict | str | None,
    parent: object | None = None,
) -> NMStatsFunc | None:
    """Create an NMStatsFunc from a dict or string name.

    Args:
        d: Dict with at least a "name" key, a string func name, or None.
        parent: Optional parent reference.

    Returns:
        An NMStatsFunc instance, or None.
    """
    if d is None:
        return None
    if isinstance(d, str):
        d = {"name": d}
    if not isinstance(d, dict):
        raise TypeError(nmu.type_error_str(d, "func", "dictionary or string"))
    if len(d) == 0:
        return None
    if "name" not in d:
        raise KeyError("missing func key 'name'")
    name = d["name"]
    if name is None:
        return None
    if not isinstance(name, str):
        raise TypeError(nmu.type_error_str(name, "func_name", "string"))
    f = name.lower()
    if f not in _STATS_FUNC_REGISTRY:
        raise ValueError("func_name: %s" % name)
    cls = _STATS_FUNC_REGISTRY[f]
    # Level dispatch: redirect to NMStatsFuncLevelNstd when "n_std" present
    if cls is NMStatsFuncLevel:
        lower_keys = {k.lower() for k in d if k.lower() != "name"}
        if "n_std" in lower_keys and "ylevel" in lower_keys:
            raise KeyError("either 'ylevel' or 'n_std' is allowed, not both")
        if "n_std" in lower_keys:
            cls = NMStatsFuncLevelNstd
        elif not lower_keys:
            raise KeyError("missing func key 'ylevel' or 'n_std'")
    # Extract valid parameter keys for this class, reject unknown keys
    kwargs: dict[str, Any] = {}
    for key, v in d.items():
        k = key.lower()
        if k == "name":
            continue
        elif k in cls._VALID_KEYS:
            kwargs[k] = v
        else:
            if cls._VALID_KEYS:
                raise KeyError("unknown func key '%s'" % key)
            else:
                raise KeyError(
                    "unknown key parameter '%s' for func '%s'" % (key, name))
    return cls(f, parent=parent, **kwargs)


class NMToolStats(NMTool):
    """
    NM Stats Tool class
    """

    def __init__(self) -> None:

        self.__win_container = NMStatsWinContainer(parent=self)
        self.__win_container.new()

        self.__xclip = True
        # if x0|x1 OOB, clip to data x-scale limits
        # if x0 = -math.inf, then x0 will be clipped to smallest x-value
        # if x1 = math.inf, then x1 will be clipped to largest x-value

        self.__ignore_nans = True
        # NumPy array analysis
        # for example: if ignore_nans,
        # then np.nanmean(array) else np.mean(array)

        self.__results: dict[str, list[Any]] = {}
        # {"w0": [ [{}, {}], [{}, {}]... ],  stats win0
        #  "w1": [ [{}, {}], [{}, {}]... ],  stats win1
        #  ...}
        # for each stats window (e.g. "w0") there is a list
        # containing results [{}, {}] for each data array.
        # for each data array there is a list [{}, {}] containing
        # results for each measure {} made for the stats window
        # e.g. baseline, main, p0, p1, slope, etc.

        self.__save_history = False
        self.__save_cache = True
        self.__save_numpy = False

    @property
    def windows(self) -> NMStatsWinContainer:
        return self.__win_container

    @property
    def xclip(self) -> bool:
        return self.__xclip

    @xclip.setter
    def xclip(self, xclip: bool) -> None:
        return self._xclip_set(xclip)

    def _xclip_set(
        self,
        xclip: bool,
        quiet: bool = nmp.QUIET
    ) -> None:
        if isinstance(xclip, bool):
            self.__xclip = xclip
        else:
            e = nmu.type_error_str(xclip, "xclip", "boolean")
            raise TypeError(e)
        nmh.history("set xclip=%s" % xclip, quiet=quiet)

    @property
    def ignore_nans(self) -> bool:
        return self.__ignore_nans

    @ignore_nans.setter
    def ignore_nans(self, ignore_nans: bool) -> None:
        return self._ignore_nans_set(ignore_nans)

    def _ignore_nans_set(
        self,
        ignore_nans: bool,
        quiet: bool = nmp.QUIET
    ) -> None:
        if isinstance(ignore_nans, bool):
            self.__ignore_nans = ignore_nans
        else:
            e = nmu.type_error_str(ignore_nans, "ignore_nans", "boolean")
            raise TypeError(e)
        nmh.history("set ignore_nans=%s" % ignore_nans, quiet=quiet)

    @property
    def save_history(self) -> bool:
        return self.__save_history

    @save_history.setter
    def save_history(self, value: bool) -> None:
        self._save_history_set(value)

    def _save_history_set(
        self,
        value: bool,
        quiet: bool = nmp.QUIET,
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "save_history", "boolean"))
        self.__save_history = value
        nmh.history("set save_history=%s" % value, quiet=quiet)

    @property
    def save_cache(self) -> bool:
        return self.__save_cache

    @save_cache.setter
    def save_cache(self, value: bool) -> None:
        self._save_cache_set(value)

    def _save_cache_set(
        self,
        value: bool,
        quiet: bool = nmp.QUIET,
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "save_cache", "boolean"))
        self.__save_cache = value
        nmh.history("set save_cache=%s" % value, quiet=quiet)

    @property
    def save_numpy(self) -> bool:
        return self.__save_numpy

    @save_numpy.setter
    def save_numpy(self, value: bool) -> None:
        self._save_numpy_set(value)

    def _save_numpy_set(
        self,
        value: bool,
        quiet: bool = nmp.QUIET,
    ) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "save_numpy", "boolean"))
        self.__save_numpy = value
        nmh.history("set save_numpy=%s" % value, quiet=quiet)

    # override, no super
    def run_init(self) -> bool:
        if isinstance(self.__results, dict):
            self.__results.clear()
        return True  # ok

    # override, no super
    def run(self) -> bool:
        if not isinstance(self.data, NMData):
            raise RuntimeError("no data selected")
        for w in self.windows:
            self.windows.selected_name = w.name
            if not w.on:
                continue
            w.compute(self.data, xclip=self.__xclip,
                      ignore_nans=self.__ignore_nans)
            # results saved to w.results
            if not w.results:
                continue
            if w.name in self.__results:
                self.__results[w.name].append(w.results)
            else:
                self.__results[w.name] = [w.results]
        return True  # ok

    # override, no super
    def run_finish(self) -> bool:
        if self.__save_history:
            self._save_history()
        if self.__save_cache:
            self._save_cache()
        if self.__save_numpy:
            self._save_numpy()
        return True  # ok

    def _save_history(self, quiet: bool = False) -> None:
        if not isinstance(self.__results, dict):
            return None
        for kwin, vlist in self.__results.items():  # windows
            nmh.history(
                "stats results for win '%s':" % kwin,
                quiet=quiet,
            )
            if not isinstance(vlist, list):
                return None
            for ilist in vlist:  # NMData
                if not isinstance(ilist, list):
                    return None
                for rdict in ilist:  # stats results
                    nmh.history(str(rdict), quiet=quiet)
        return None

    def _save_cache(self) -> int | None:
        if not isinstance(self.folder, NMFolder):
            return None
        if not self.__results:
            raise RuntimeError("there are no results to save")
        return self.folder.toolresults_save("stats", self.__results)

    # Numeric keys extracted from result dicts into NMData arrays.
    # Maps result dict key → NMData name suffix and units source key.
    _NUMERIC_KEYS: dict[str, tuple[str, str | None]] = {
        "s":    ("s",    "sunits"),
        "x":    ("x",    "xunits"),
        "i":    ("i",    None),
        "i0":   ("i0",   None),
        "i1":   ("i1",   None),
        "n":    ("n",    None),
        "nans": ("nans", None),
        "infs": ("infs", None),
        "var":  ("var",  "sunits"),
        "std":  ("std",  "sunits"),
        "sem":  ("sem",  "sunits"),
        "b":    ("b",    "bunits"),
        "dx":   ("dx",   "xunits"),
    }

    def _save_numpy(self) -> NMToolFolder | None:
        if not isinstance(self.folder, NMFolder):
            return None
        if not self.__results:
            raise RuntimeError("there are no results to save")

        # Find next unused folder name stats0, stats1, ...
        tf = self.folder.toolfolder
        i = 0
        f = None
        while f is None:
            try:
                f = tf.new(name="stats%d" % i)
            except KeyError:
                i += 1

        for wname, vlist in self.__results.items():
            # vlist: list of lists, one inner list per data wave
            # Each inner list contains result dicts for that wave

            # Collect data path strings (one per wave, from first rdict)
            data_paths: list[str] = []
            # Collect result dicts grouped by id: {id_str: [rdict, ...]}
            id_rdicts: dict[str, list[dict]] = {}

            for ilist in vlist:
                wave_path: str | None = None
                for rdict in ilist:
                    id_str = rdict.get("id", "main")
                    if wave_path is None:
                        wave_path = rdict.get("data", "")
                    if id_str not in id_rdicts:
                        id_rdicts[id_str] = []
                    id_rdicts[id_str].append(rdict)
                if wave_path is not None:
                    data_paths.append(wave_path)

            # Save data path strings
            if data_paths and f.data is not None:
                f.data.new(
                    "ST_%s_data" % wname,
                    nparray=np.array(data_paths, dtype=object),
                )

            # Save numeric arrays per id
            for id_str, rdicts in id_rdicts.items():
                for rkey, (suffix, units_key) in self._NUMERIC_KEYS.items():
                    values = [rdict.get(rkey) for rdict in rdicts]
                    if all(v is None for v in values):
                        continue  # key not present for this func
                    arr = np.array(
                        [v if v is not None else math.nan for v in values],
                        dtype=float,
                    )
                    units = rdicts[0].get(units_key) if units_key else None
                    yscale = {"units": units} if units else None
                    dname = "ST_%s_%s_%s" % (wname, id_str, suffix)
                    if f.data is not None:
                        f.data.new(dname, nparray=arr, yscale=yscale)

                # Save warnings if any occurred
                warnings = [rdict.get("warning") for rdict in rdicts]
                if any(w is not None for w in warnings):
                    dname = "ST_%s_%s_warning" % (wname, id_str)
                    if f.data is not None:
                        f.data.new(
                            dname,
                            nparray=np.array(
                                [w or "" for w in warnings], dtype=object
                            ),
                        )

        return f

    """
    def results_save(self) -> bool:
        r = {}
        r["tool"] = "stats"
        r["date"] = str(datetime.datetime.now())
        r["results"] = self.__results
        for i in range(99):
            sname = "stats" + str(i)
            if sname not in self.folder.toolresults:
                self.folder.toolresults[sname] = r
                break
        print(self.folder.toolresults)
        fname = "stats_test"
        f = self.folder.toolfolder.new(fname)
        f.data.new("ST_w0_avg_")
        "ST_" + w.name + func_name
        return True
    """


class NMStatsWin:
    """NM Stats Window class.

    Lightweight class (does not inherit NMObject) following the NMScaleY
    pattern. Each window defines a stats measurement with x-range, function,
    optional baseline, and optional transforms.
    """

    def __init__(
        self,
        parent: object | None = None,
        name: str = "NMStatsWin0",
        win: dict[str, object] | None = None,
    ) -> None:
        self._parent = parent
        if not isinstance(name, str):
            raise TypeError(nmu.type_error_str(name, "name", "string"))
        if not name or not nmu.name_ok(name):
            raise ValueError("name: %s" % name)
        self._name = name

        self.__on = True
        self.__func: NMStatsFunc | None = None
        self.__x0 = -math.inf
        self.__x1 = math.inf
        self.__transform: list[NMTransform] | None = None
        self.__results: list[dict[str, Any]] = []  # [ {}, {} ...] list of dictionaries

        # baseline
        self.__bsln_on = False
        self.__bsln_func: dict[str, Any] = {}
        self.__bsln_x0 = -math.inf
        self.__bsln_x1 = math.inf

        if win is None:
            pass  # ok
        elif isinstance(win, dict):
            self._win_set(win, quiet=True)
        else:
            e = nmu.type_error_str(win, "win", "dictionary")
            raise TypeError(e)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMStatsWin):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __deepcopy__(self, memo: dict) -> NMStatsWin:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for attr, value in self.__dict__.items():
            if attr == "_parent":
                setattr(result, attr, None)
            else:
                setattr(result, attr, copy.deepcopy(value, memo))
        return result

    @property
    def name(self) -> str:
        return self._name

    def copy(self) -> NMStatsWin:
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        """Serialize this stats window to a dict."""
        if self.__transform is not None:
            transform_dicts = [t.to_dict() for t in self.__transform]
        else:
            transform_dicts = None
        return {
            "name": self._name,
            "on": self.__on,
            "func": self.__func.to_dict() if self.__func else {},
            "x0": self.__x0,
            "x1": self.__x1,
            "transform": transform_dicts,
            "bsln_on": self.__bsln_on,
            "bsln_func": self.__bsln_func,
            "bsln_x0": self.__bsln_x0,
            "bsln_x1": self.__bsln_x1
        }

    def _win_set(
        self,
        win: dict[str, object],
        quiet: bool = nmp.QUIET
    ) -> None:
        if not isinstance(win, dict):
            e = nmu.type_error_str(win, "win", "dictionary")
            raise TypeError(e)
        for k, v in win.items():
            if not isinstance(k, str):
                e = nmu.type_error_str(k, "key", "string")
                raise TypeError(e)
            k = k.lower()
            if k == "name":
                continue  # name is set via constructor, not _win_set
            elif k == "on":
                self._on_set(v, quiet=True)  # type: ignore[arg-type]
            elif k == "func":
                self._func_set(v, quiet=True)  # type: ignore[arg-type]
            elif k == "x0":
                self._x_set("x0", v, quiet=True)  # type: ignore[arg-type]
            elif k == "x1":
                self._x_set("x1", v, quiet=True)  # type: ignore[arg-type]
            elif k == "transform":
                self._transform_set(v, quiet=True)  # type: ignore[arg-type]
            elif k == "bsln_on":
                self._bsln_on_set(v, quiet=True)  # type: ignore[arg-type]
            elif k == "bsln_func":
                self._bsln_func_set(v, quiet=True)  # type: ignore[arg-type]
            elif k == "bsln_x0":
                self._x_set("bsln_x0", v, quiet=True)  # type: ignore[arg-type]
            elif k == "bsln_x1":
                self._x_set("bsln_x1", v, quiet=True)  # type: ignore[arg-type]
            else:
                raise KeyError("unknown key '%s'" % k)
        nmh.history("set win=%s" % win, path=self._name, quiet=quiet)
        return None

    @property
    def on(self) -> bool:
        return self.__on

    @on.setter
    def on(self, on: bool) -> None:
        return self._on_set(on)

    def _on_set(self, on: bool, quiet: bool = nmp.QUIET) -> None:
        if not isinstance(on, bool):
            e = nmu.type_error_str(on, "on", "boolean")
            raise TypeError(e)
        self.__on = on
        nmh.history("set on=%s" % on, path=self._name, quiet=quiet)
        return None

    @property
    def func(self) -> dict:
        return self.__func.to_dict() if self.__func else {}

    @func.setter
    def func(self, func: dict | str) -> None:
        self._func_set(func)
        return None

    def _func_set(
        self,
        func: dict | str | None,
        quiet: bool = nmp.QUIET
    ) -> None:
        if func is None or (isinstance(func, dict) and len(func) == 0):
            self.__func = None
            return None
        if isinstance(func, dict) and "name" not in func:
            if self.__func is not None:
                # allows updating func parameters without passing name
                func["name"] = self.__func.name
            else:
                raise KeyError("missing func key 'name'")
        self.__func = _stats_func_from_dict(func, parent=self._parent)
        if self.__func is not None:
            nmh.history(
                "set func=%s" % self.__func.to_dict(),
                path=self._name,
                quiet=quiet,
            )
        return None

    @property
    def x0(self) -> float:
        return self.__x0

    @x0.setter
    def x0(self, x0: float) -> None:
        return self._x_set("x0", x0)

    def _x_set(
        self,
        xname: str,  # e.g. "x0" or "bsln_x0"
        x: float,
        quiet: bool = nmp.QUIET
    ) -> None:
        x = float(x)  # might raise type error
        if math.isnan(x):
            raise ValueError(xname + ": %s" % x)
        n = xname.lower()
        if n not in ("x0", "x1", "bsln_x0", "bsln_x1"):
            raise ValueError("xname: %s" % xname)
        if math.isinf(x):
            x = -math.inf if n.endswith("x0") else math.inf
        if n == "x0":
            self.__x0 = x
        elif n == "x1":
            self.__x1 = x
        elif n == "bsln_x0":
            self.__bsln_x0 = x
        elif n == "bsln_x1":
            self.__bsln_x1 = x
        nmh.history("set %s=%s" % (n, x), path=self._name, quiet=quiet)
        return None

    @property
    def x1(self) -> float:
        return self.__x1

    @x1.setter
    def x1(self, x1: float) -> None:
        return self._x_set("x1", x1)

    @property
    def transform(self) -> list[NMTransform] | None:
        return self.__transform

    @transform.setter
    def transform(self, transform_list: list) -> None:
        return self._transform_set(transform_list)

    def _transform_set(
        self,
        transform_list: list[NMTransform] | list[dict] | None,
        quiet: bool = nmp.QUIET
    ) -> None:
        if transform_list is None:
            self.__transform = None
            nmh.history("set transform=None", path=self._name, quiet=quiet)
            return None
        if not isinstance(transform_list, list):
            e = nmu.type_error_str(transform_list, "transform_list", "list")
            raise TypeError(e)
        result = []
        for item in transform_list:
            if isinstance(item, NMTransform):
                result.append(item)
            elif isinstance(item, dict):
                result.append(_transform_from_dict(item))
            else:
                e = nmu.type_error_str(
                    item, "transform item", "NMTransform or dict"
                )
                raise TypeError(e)
        self.__transform = result
        nmh.history(
            "set transform=[%s]" % ", ".join(t.type_str for t in result),
            path=self._name,
            quiet=quiet,
        )
        return None

    @property
    def bsln_on(self) -> bool:
        return self.__bsln_on

    @bsln_on.setter
    def bsln_on(self, on: bool) -> None:
        return self._bsln_on_set(on)

    def _bsln_on_set(self, on: bool, quiet: bool = nmp.QUIET) -> None:
        if not isinstance(on, bool):
            e = nmu.type_error_str(on, "on", "boolean")
            raise TypeError(e)
        self.__bsln_on = on
        nmh.history("set bsln_on=%s" % on, path=self._name, quiet=quiet)
        return None

    @property
    def bsln_func(self) -> dict:
        return self.__bsln_func

    @bsln_func.setter
    def bsln_func(self, func: dict | str) -> None:
        self._bsln_func_set(func)
        return None

    def _bsln_func_set(
        self,
        func: dict | str | None,
        quiet: bool = nmp.QUIET
    ) -> None:

        if func is None:
            self.__bsln_func.clear()
            return None
        if isinstance(func, dict):
            if len(func) == 0:
                self.__bsln_func.clear()
                return None
            if "name" not in func:
                e = "missing func key 'name'"
                raise KeyError(e)
            func_name = func["name"]
        elif isinstance(func, str):
            func_name = func
        else:
            e = nmu.type_error_str(func, "func", "dictionary, string or None")
            raise TypeError(e)

        if func_name is None:
            self.__bsln_func.clear()
            return None
        if not isinstance(func_name, str):
            e = nmu.type_error_str(func_name, "func_name", "string")
            raise TypeError(e)

        found = False
        for f in FUNC_NAMES_BSLN:
            if f.lower() == func_name.lower():
                found = True
                break
        if not found:
            raise ValueError("func_name: %s" % func_name)

        self.__bsln_func.clear()
        self.__bsln_func.update({"name": func_name.lower()})
        nmh.history(
            "set bsln_func=%s" % self.__bsln_func,
            path=self._name,
            quiet=quiet,
        )
        return None

    @property
    def bsln_x0(self) -> float:
        return self.__bsln_x0

    @bsln_x0.setter
    def bsln_x0(self, x0: float) -> None:
        return self._x_set("bsln_x0", x0)

    @property
    def bsln_x1(self) -> float:
        return self.__bsln_x1

    @bsln_x1.setter
    def bsln_x1(self, x1: float) -> None:
        return self._x_set("bsln_x1", x1)

    @property
    def results(self) -> list[dict]:
        return self.__results

    def _run_stat(self, data, func, id_str, x0, x1, xclip, ignore_nans,
                  **extra):
        """Create a result dict, append to results, and call stats()."""
        r: dict[str, Any] = {"win": self.name, "id": id_str}
        r.update(extra)
        r["func"] = func
        r["x0"] = x0
        r["x1"] = x1
        self.__results.append(r)
        stats(data, func, x0=x0, x1=x1, xclip=xclip,
              ignore_nans=ignore_nans, results=r)
        return r

    def compute(
        self,
        data: NMData,
        xclip: bool = False,  # if x0|x1 OOB, clip to data x-scale limits
        ignore_nans: bool = False,
        quiet: bool = nmp.QUIET
    ) -> list:

        # Apply transforms to a copy of the data (original never mutated)
        if self.__transform and len(self.__transform) > 0:
            if data is not None and data.nparray is not None:
                transformed = apply_transforms(
                    data.nparray, self.__transform, xscale=data.xscale
                )
                data = NMData(
                    name=data.name,
                    nparray=transformed,
                    xscale=data.xscale.to_dict(),
                    yscale=data.yscale.to_dict(),
                )

        self.__results = []

        if data is None or self.__func is None:
            return self.__results

        if not isinstance(xclip, bool):
            xclip = False

        if not isinstance(ignore_nans, bool):
            ignore_nans = True

        bsln_result: dict[str, Any] = {}

        if self.__bsln_on:
            self.__func.validate_baseline(self.__bsln_func.get("name"))
            bsln_result = self._run_stat(
                data, self.__bsln_func.copy(), "bsln",
                self.__bsln_x0, self.__bsln_x1, xclip, ignore_nans
            )
        elif self.__func.needs_baseline:
            raise RuntimeError(
                "func '%s' requires baseline" % self.__func.name
            )

        self.__func.compute(
            data, self.__x0, self.__x1, xclip, ignore_nans,
            self._run_stat, bsln_result
        )

        nmh.history(
            "compute func=%s, x0=%s, x1=%s, n=%d"
            % (self.__func.name, self.__x0, self.__x1, len(self.__results)),
            path=self._name,
            quiet=quiet,
        )

        return self.__results


class NMStatsWinContainer:
    """Simple container of NMStatsWin objects with auto-naming."""

    def __init__(
        self,
        parent: object | None = None,
        name_prefix: str = "w",
    ) -> None:
        self._parent = parent
        self._prefix = name_prefix
        self._windows: dict[str, NMStatsWin] = {}
        self._count = 0
        self.selected_name: str | None = None

    def new(self, quiet: bool = nmp.QUIET) -> NMStatsWin:
        name = "%s%d" % (self._prefix, self._count)
        self._count += 1
        w = NMStatsWin(parent=self._parent, name=name)
        self._windows[name] = w
        if self.selected_name is None:
            self.selected_name = name
        nmh.history("new NMStatsWin=%s" % name, quiet=quiet)
        return w

    def __iter__(self):
        return iter(self._windows.values())

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, name: str) -> NMStatsWin:
        return self._windows[name]

    def __contains__(self, name: str) -> bool:
        return name in self._windows


# =========================================================================
# stats() dispatch handlers
# =========================================================================


def _stat_maxmin(f, func, yarray, data, i0, ysize, ignore_nans, results,
                 yunits, **_):
    if "max" in f:
        index = np.nanargmax(yarray) if ignore_nans else np.argmax(yarray)
    else:
        index = np.nanargmin(yarray) if ignore_nans else np.argmin(yarray)
    results["s"] = yarray[index]
    results["sunits"] = yunits
    i = int(index) + int(i0)  # shift due to slicing
    results["i"] = i
    results["x"] = data.get_xvalue(i)
    results["xunits"] = data.xscale.units

    n_avg = 0
    if "n_avg" in func and 0 <= i < ysize:
        n_avg = int(func["n_avg"])
        if n_avg <= 1:
            n_avg = 0

    if n_avg > 1:
        if n_avg % 2 == 0:  # even
            i0_m = int(i - 0.5 * n_avg)
            i1_m = int(i0_m + n_avg - 1)
        else:  # odd
            i0_m = int(i - 0.5 * (n_avg - 1))
            i1_m = int(i + 0.5 * (n_avg - 1))
        i0_m = max(i0_m, 0)
        i0_m = min(i0_m, ysize - 1)
        i1_m = max(i1_m, 0)
        i1_m = min(i1_m, ysize - 1)
        if i0_m == 0 and i1_m == ysize - 1:
            yarr = data.nparray
        else:
            yarr = data.nparray[i0_m:i1_m+1]
        results["s"] = np.nanmean(yarr) if ignore_nans else np.mean(yarr)

    return results


def _stat_level(f, func, yarray, data, i0, ignore_nans, results, yunits,
                xunits, found_xarray, xarray=None, xstart=None, **_):
    if "ylevel" in func:
        ylevel = func["ylevel"]
    else:
        raise KeyError("missing key 'ylevel'")
    if found_xarray:
        i_x = find_level_crossings(
            yarray, ylevel, func_name=f,
            xarray=xarray, ignore_nans=ignore_nans
        )
    else:
        xstart_val = xstart if isinstance(xstart, float) else 0.0
        xdelta_val = float(data.xscale.delta)
        i_x = find_level_crossings(
            yarray, ylevel, func_name=f,
            xstart=xstart_val, xdelta=xdelta_val,
            ignore_nans=ignore_nans
        )
    indexes = i_x[0]
    xvalues = i_x[1]
    if "func" in results:
        fxn = results["func"]
        if isinstance(fxn, dict):
            fxn.update({"yunits": yunits})
    if indexes.size > 0:  # return first level crossing
        results["i"] = indexes[0] + i0  # shift due to slicing
        results["x"] = xvalues[0]  # shift not needed for x-values
        results["xunits"] = xunits
    else:
        results["i"] = None
        results["x"] = None
    return results


def _stat_slope(yarray, data, ignore_nans, results, yunits, xunits,
                found_xarray, xarray=None, xstart=None, **_):
    if found_xarray:
        mb = linear_regression(
            yarray, xarray=xarray, ignore_nans=ignore_nans
        )
    else:
        xstart_val = xstart if isinstance(xstart, float) else 0.0
        xdelta_val = float(data.xscale.delta)
        mb = linear_regression(
            yarray, xstart=xstart_val, xdelta=xdelta_val,
            ignore_nans=ignore_nans
        )
    if mb:
        results["s"] = mb[0]
        if isinstance(xunits, str) and isinstance(yunits, str):
            results["sunits"] = yunits + "/" + xunits
        else:
            results["sunits"] = None
        results["b"] = mb[1]
        if isinstance(yunits, str):
            results["bunits"] = yunits
        else:
            results["bunits"] = None
    else:
        results["s"] = None
        results["b"] = None
    return results


def _stat_median(yarray, ignore_nans, results, yunits, **_):
    results["s"] = np.nanmedian(yarray) if ignore_nans else np.median(yarray)
    results["sunits"] = yunits
    return results


def _stat_mean(f, yarray, ignore_nans, results, yunits, n, **_):
    results["s"] = np.nanmean(yarray) if ignore_nans else np.mean(yarray)
    results["sunits"] = yunits
    if "+var" in f:
        results["var"] = np.nanvar(yarray) if ignore_nans else np.var(yarray)
    if "+std" in f:
        results["std"] = np.nanstd(yarray) if ignore_nans else np.std(yarray)
    if "+sem" in f:
        std = np.nanstd(yarray) if ignore_nans else np.std(yarray)
        results["sem"] = std / math.sqrt(n)
    return results


def _stat_var(yarray, ignore_nans, results, yunits, **_):
    results["s"] = np.nanvar(yarray) if ignore_nans else np.var(yarray)
    if isinstance(yunits, str):
        results["sunits"] = yunits + "**2"
    else:
        results["sunits"] = None
    return results


def _stat_std(yarray, ignore_nans, results, yunits, **_):
    results["s"] = np.nanstd(yarray) if ignore_nans else np.std(yarray)
    results["sunits"] = yunits
    return results


def _stat_sem(yarray, ignore_nans, results, yunits, n, **_):
    std = np.nanstd(yarray) if ignore_nans else np.std(yarray)
    results["s"] = std / math.sqrt(n)
    results["sunits"] = yunits
    return results


def _stat_rms(yarray, ignore_nans, results, yunits, n, **_):
    sos = np.nansum(np.square(yarray)) if ignore_nans else np.sum(
        np.square(yarray))
    results["s"] = math.sqrt(sos / n)
    results["sunits"] = yunits
    return results


def _stat_sum(yarray, ignore_nans, results, yunits, **_):
    results["s"] = np.nansum(yarray) if ignore_nans else np.sum(yarray)
    results["sunits"] = yunits
    return results


def _stat_pathlength(yarray, data, ignore_nans, results, yunits, xunits,
                     found_xarray, xarray=None, **_):
    w = None
    if isinstance(xunits, str) and isinstance(yunits, str):
        if xunits != yunits:
            raise ValueError(
                "pathlength: x- and y-scales have "
                + "different units: %s != %s" % (xunits, yunits))
    else:
        w = "pathlength assumes x- and y-scales have the same units"
    if found_xarray:
        dx2 = np.square(np.diff(xarray))
        dy2 = np.square(np.diff(yarray))
        h = np.sqrt(np.add(dx2, dy2))
    else:
        dx = float(data.xscale.delta)
        dx2 = dx**2
        dy2 = np.square(np.diff(yarray))
        h = np.sqrt(dx2 + dy2)
    results["s"] = np.nansum(h) if ignore_nans else np.sum(h)
    results["sunits"] = yunits
    if w:
        results["warning"] = w
    return results


def _stat_area(yarray, data, ignore_nans, results, yunits, xunits,
               found_xarray, xarray=None, **_):
    if found_xarray:
        if ignore_nans:
            results["s"] = np.nansum(np.multiply(xarray, yarray))
        else:
            results["s"] = np.sum(np.multiply(xarray, yarray))
    else:
        sum_y = np.nansum(yarray) if ignore_nans else np.sum(yarray)
        results["s"] = sum_y * data.xscale.delta
    if isinstance(xunits, str) and isinstance(yunits, str):
        if xunits == yunits:
            results["sunits"] = xunits + "**2"
        else:
            results["sunits"] = xunits + "*" + yunits
    else:
        results["sunits"] = None
    return results


def _stat_count(results, **_):
    return results


_STATS_DISPATCH = {
    "max": _stat_maxmin,
    "min": _stat_maxmin,
    "mean@max": _stat_maxmin,
    "mean@min": _stat_maxmin,
    "level": _stat_level,
    "level+": _stat_level,
    "level-": _stat_level,
    "slope": _stat_slope,
    "median": _stat_median,
    "mean": _stat_mean,
    "mean+var": _stat_mean,
    "mean+std": _stat_mean,
    "mean+sem": _stat_mean,
    "var": _stat_var,
    "std": _stat_std,
    "sem": _stat_sem,
    "rms": _stat_rms,
    "sum": _stat_sum,
    "pathlength": _stat_pathlength,
    "area": _stat_area,
    "count": _stat_count,
    "count_nans": _stat_count,
    "count_infs": _stat_count,
}


def stats(
    data: NMData,
    func: dict,
    x0: float = -math.inf,
    x1: float = math.inf,  # math.inf denotes xclip = True
    xclip: bool = False,  # if x0|x1 OOB, clip to data x-scale limits
    ignore_nans: bool = False,
    results: dict | None = None
) -> dict:  # returns results

    if not isinstance(data, NMData):
        e = nmu.type_error_str(data, "data", "NMData")
        raise TypeError(e)
    if not isinstance(data.nparray, np.ndarray):
        e = nmu.type_error_str(data.nparray, "nparray", "NumPy.ndarray")
        raise TypeError(e)

    if not isinstance(func, dict):
        e = nmu.type_error_str(func, "func", "dictionary")
        raise TypeError(e)
    if "name" not in func:
        e = "missing key 'name' in func dictionary"
        raise KeyError(e)

    f = func["name"]
    if not isinstance(f, str):
        e = nmu.type_error_str(f, "func_name", "string")
        raise TypeError(e)
    f = f.lower()

    found_xarray = isinstance(data.xarray, np.ndarray)
    ysize = data.nparray.size

    if found_xarray and data.xarray.size != ysize:
        e = ("x-y paired NumPy arrays have different size: %s != %s"
             % (data.xarray.size, ysize))
        raise ValueError(e)

    if results is None:
        results = {}
    elif not isinstance(results, dict):
        e = nmu.type_error_str(results, "results", "dictionary")
        raise TypeError(e)

    # results["func"] = f
    results["data"] = data.path_str

    xunits = data.xscale.units
    yunits = data.yscale.units

    i0 = data.get_xindex(x0, clip=xclip)
    i1 = data.get_xindex(x1, clip=xclip)

    results["i0"] = i0
    results["i1"] = i1

    if i0 is None:
        e = "failed to compute i0 from x0"
        # raise ValueError(e)
        results["error"] = e
        return results
    if i1 is None:
        e = "failed to compute i1 from x1"
        # raise ValueError(e)
        results["error"] = e
        return results

    if f == "value@x0":
        results["s"] = data.nparray[i0]
        results["sunits"] = yunits
        return results
    if f == "value@x1":
        results["s"] = data.nparray[i1]
        results["sunits"] = yunits
        return results

    # if i0 == i1:  # 1-point array should be ok
        # e = "i0 = i1: %s = %s" % (i0, i1)
        # raise ValueError("i0 = i1: %s = %s" % (i0, i1))
        # results["error"] = e
        # return results
    if i0 > i1:  # switch
        isave = i0
        i0 = i1
        i1 = isave
        results["i0"] = i0
        results["i1"] = i1

    if i0 == 0 and i1 == ysize - 1:
        yarray = data.nparray
        if found_xarray:
            xarray = data.xarray
        else:
            xstart = data.xscale.start
    else:  # slice
        yarray = data.nparray[i0:i1+1]
        if found_xarray:
            xarray = data.xarray[i0:i1+1]
        else:
            xstart = data.get_xvalue(i0)

    nans = np.count_nonzero(np.isnan(yarray))
    infs = np.count_nonzero(np.isinf(yarray))
    if ignore_nans:
        n = yarray.size - nans
    else:
        n = yarray.size
    results["n"] = n
    results["nans"] = nans
    results["infs"] = infs

    ctx = {
        "f": f, "func": func, "yarray": yarray, "data": data,
        "i0": i0, "ysize": ysize, "ignore_nans": ignore_nans,
        "results": results, "yunits": yunits, "xunits": xunits, "n": n,
        "found_xarray": found_xarray,
    }
    if found_xarray:
        ctx["xarray"] = xarray
    else:
        ctx["xstart"] = xstart

    handler = _STATS_DISPATCH.get(f)
    if handler is None:
        raise ValueError("unknown function '%s'" % func)

    return handler(**ctx)


def find_level_crossings(
    yarray,
    ylevel: float,  # the y-axis level (yl) to search for
    func_name: str = "level",
    # "level":  find all level crossings (both pos and neg slopes)
    # "level+": find level crossings on positive slopes
    # "level-": find level crossings on negative slopes
    xarray=None,  # NumPy array containing x-scale
    xstart: float = 0,  # x-scale start value, used if xarray=None
    xdelta: float = 1,  # x-scale delta increment, used if xarray=None
    i_nearest: bool = True,
    # return array indexes (i-values) that are nearest to ylevel crossings
    # method uses linear interpoloation
    # otherwise returns array indexes immediately after level crossing (i1)
    x_interp: bool = True,
    # return estimated x-value at location of ylevel crossing
    # method uses linear interpoloation
    # otherwise returns x-values at corresponding i-values (e.g. x1 at i1)
    ignore_nans: bool = True
) -> tuple:
    #                         y y y y y
    #                    y1 y
    #
    #  yl-->
    #        y y y y y y0
    #
    #  y y y
    #
    #  i 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5
    #  x 0 2 4 6 8 0 2 4 6 8 0 2 4 6 8 0 (x0 = 0, dx = 2)
    #  results:
    #    y0, i0=7, x0=14   before y-level crossing
    #    y1, i1=8, x1=16   after y-level crossing
    #    i-nearest = 8
    #    x-interp = 14.3333

    if not isinstance(func_name, str):
        e = nmu.type_error_str(func_name, "func_name", "string")
        raise TypeError(e)

    f = func_name.lower()
    if "level" not in f:
        e = "func_name: '%s'" % func_name
        raise ValueError(e)

    if isinstance(ylevel, float):
        if math.isinf(ylevel) or math.isnan(ylevel):
            raise ValueError("ylevel: '%s'" % ylevel)
    else:
        ylevel = float(ylevel)  # might raise type error

    if not isinstance(yarray, np.ndarray):
        e = nmu.type_error_str(yarray, "yarray", "NumPy.ndarray")
        raise TypeError(e)

    found_xarray = False

    if xarray is None:
        pass
    elif isinstance(xarray, np.ndarray):
        if xarray.size == yarray.size:
            found_xarray = True
        else:
            e = ("x-y paired NumPy arrays have different size: %s != %s"
                 % (xarray.size, yarray.size))
            raise ValueError(e)
    else:
        e = nmu.type_error_str(xarray, "xarray", "NumPy.ndarray")
        raise TypeError(e)

    if isinstance(xstart, float):
        if math.isinf(xstart) or math.isnan(xstart):
            raise ValueError("xstart: '%s'" % xstart)
    else:
        xstart = float(xstart)  # might raise type error

    if isinstance(xdelta, float):
        if math.isinf(xdelta) or math.isnan(xdelta):
            raise ValueError("xdelta: '%s'" % xdelta)
    else:
        xdelta = float(xdelta)  # might raise type error

    if not isinstance(i_nearest, bool):
        i_nearest = True

    if not isinstance(x_interp, bool):
        x_interp = True

    if not isinstance(ignore_nans, bool):
        ignore_nans = True

    level_crossings = np.diff(yarray > ylevel, prepend=False)
    # diff output array is one point smaller than yarray
    # so prepend 1 False to beginning
    # True denotes transition through ylevel, either pos or neg direction
    # RuntimeWarning: invalid value encountered in greater
    # warning due to NaNs
    """
    with warnings.catch_warnings(record=True) as wlist:
        level_crossings = np.diff(yarray > ylevel, prepend=False)
        for w in wlist:
            print(w)
            """
    locations = np.argwhere(level_crossings)  # grab True locations
    if len(locations.shape) != 2:  # shape = (N, 1)
        raise RuntimeError("locations shape should be 2")
    locations = locations[:, 0]
    indexes = []
    xvalues = []

    for i in locations:

        if i == 0:
            continue
            # e = "location index should be great than 0: %s" % i
            # raise ValueError(e)
            # should not occur since False is prepended to level_crossings

        # transition occurs between y0 and y1
        y0 = yarray[i-1]
        y1 = yarray[i]

        if f == "level+":
            if y1 <= y0:
                continue  # wrong slope
        elif f == "level-":
            if y1 >= y0:
                continue  # wrong slope

        if found_xarray:
            x0 = xarray[i-1]
            x1 = xarray[i]
            dx = x1 - x0
        else:
            x0 = xstart + (i - 1) * xdelta
            x1 = xstart + i * xdelta
            dx = xdelta

        if not i_nearest:  # save index just after y-level crossings
            indexes.append(i)
            xvalues.append(x1)
            continue

        # find closest index via linear interpolation
        # compute x-location via linear interpolation (x-interp)
        dy = y1 - y0
        m = dy / dx
        b = y1 - m * x1
        x = (ylevel - b) / m

        if abs(x - x0) <= abs(x - x1):
            indexes.append(i-1)
            if x_interp:
                xvalues.append(x)
            else:
                xvalues.append(x0)
        else:
            indexes.append(i)
            if x_interp:
                xvalues.append(x)
            else:
                xvalues.append(x1)

    return (np.array(indexes), np.array(xvalues))


def xinterp(ylevel, x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    m = dy / dx
    b = y1 - m * x1
    x = (ylevel - b) / m
    return x


def linear_regression(
    yarray,
    xarray=None,
    xstart: float = 0,
    xdelta: float = 1,
    ignore_nans: bool = True
) -> tuple:  # (m, b)

    if not isinstance(yarray, np.ndarray):
        e = nmu.type_error_str(yarray, "yarray", "NumPy.ndarray")
        raise TypeError(e)

    found_xarray = False

    if xarray is None:
        pass
    elif isinstance(xarray, np.ndarray):
        if xarray.size == yarray.size:
            found_xarray = True
        else:
            e = ("x-y paired NumPy arrays have different size: %s != %s"
                 % (xarray.size, yarray.size))
            raise ValueError(e)
    else:
        e = nmu.type_error_str(xarray, "xarray", "NumPy.ndarray")
        raise TypeError(e)

    if not found_xarray:

        if isinstance(xstart, float):
            if math.isinf(xstart) or math.isnan(xstart):
                raise ValueError("xstart: '%s'" % xstart)
        elif isinstance(xstart, int) and not isinstance(xstart, bool):
            pass
        else:
            e = nmu.type_error_str(xstart, "xstart", "float")
            raise TypeError(e)

        if isinstance(xdelta, float):
            if math.isinf(xdelta) or math.isnan(xdelta):
                raise ValueError("xdelta: '%s'" % xdelta)
        elif isinstance(xdelta, int) and not isinstance(xdelta, bool):
            pass
        else:
            e = nmu.type_error_str(xdelta, "xdelta", "float")
            raise TypeError(e)

        x0 = xstart
        x1 = xstart + (yarray.size - 1) * xdelta
        xarray = np.linspace(x0, x1, yarray.size)

    if ignore_nans:
        mask = ~np.isnan(yarray)
        xarray = xarray[mask]
        yarray = yarray[mask]

    m, b = np.polyfit(xarray, yarray, deg=1)

    return (m, b)
