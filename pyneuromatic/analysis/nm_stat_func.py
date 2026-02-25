# -*- coding: utf-8 -*-
"""
NMStatFunc class hierarchy: stat function types with parameters and validation.

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
from typing import Any

import pyneuromatic.core.nm_utilities as nmu


FUNC_NAMES_BASIC = (
    "median",       # np.median
    "mean",         # np.mean (Igor: "avg")
    "mean+var",
    "mean+std",
    "mean+sem",
    "var",          # np.var
    "std",          # np.std (Igor: "sdev")
    "sem",
    "rms",
    "sum",          # np.sum
    "pathlength",
    "area",
    "slope",
    "value@x0",
    "value@x1",
    "count",        # np.size (Igor: "numpnts")
    "count_nans",
    "count_infs",
    # TODO: add "onset" — Igor version requires sigmoid curve fit
)

FUNC_NAMES_MAXMIN = (
    "max",          # np.argmax
    "min",          # np.argmin
    "mean@max",
    "mean@min",
)

FUNC_NAMES_LEVEL = ("level", "level+", "level-")

FUNC_NAMES_RISETIME = (
    "risetime+", "risetime-", "risetimeslope+", "risetimeslope-",
)

FUNC_NAMES_FALLTIME = (
    "falltime+", "falltime-", "falltimeslope+", "falltimeslope-",
)

FUNC_NAMES_DECAYTIME = ("decaytime+", "decaytime-")

FUNC_NAMES_FWHM = ("fwhm+", "fwhm-")

DECAY_TIME_DEFAULT_PCT: float = 100.0 / math.e  # ≈ 36.7879% — exact 1/e as a percentage

FUNC_NAMES_BSLN = (
    "median",
    "mean",
    "mean+var",
    "mean+std",
    "mean+sem",
)


def badvalue(n: float | None) -> bool:
    """Return True if n is None, NaN, or infinite."""
    return n is None or math.isnan(n) or math.isinf(n)


# =========================================================================
# NMStatFunc class hierarchy
# =========================================================================


class NMStatFunc:
    """Base class for stat function types.

    Lightweight class (following NMTransform pattern) that represents
    a statistics function with its parameters and compute pipeline.

    Each subclass corresponds to one category of stat function and stores
    validated parameters, exposes them via `to_dict()`, and implements the
    full measurement pipeline in `compute()`.

    Args:
        name: Stat function name string (e.g., ``"mean"``, ``"risetime+"``).
    """

    _VALID_KEYS: set[str] = set()

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        """Return string representation showing class name and parameters."""
        d = self.to_dict()
        params = ", ".join("%s=%r" % (k, v) for k, v in d.items())
        return "%s(%s)" % (self.__class__.__name__, params)

    def __eq__(self, other: object) -> bool:
        """Return True if other has the same to_dict() as this instance."""
        if isinstance(other, NMStatFunc):
            return self.to_dict() == other.to_dict()
        if isinstance(other, dict):
            return self.to_dict() == other
        return NotImplemented

    def __getitem__(self, key: str) -> object:
        """Return parameter value by key, equivalent to ``to_dict()[key]``."""
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    @property
    def name(self) -> str:
        """Stat function name string."""
        return self._name

    def to_dict(self) -> dict:
        """Return function parameters as a dict with at least a ``"name"`` key."""
        return {"name": self._name}

    @property
    def needs_baseline(self) -> bool:
        """True if this func requires a baseline result before ``compute()``."""
        return False

    def validate_baseline(self, bsln_func_name: str | None) -> None:
        """Raise RuntimeError if the baseline func is incompatible with this func.

        Args:
            bsln_func_name: The baseline func name, or None if no baseline set.
        """
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
        """Execute the stat computation, appending results via run_stat.

        Args:
            data: Input data object with x/y arrays.
            x0: Left x-boundary of the main stat window.
            x1: Right x-boundary of the main stat window.
            xclip: If True, clip data to [x0, x1] before computing.
            ignore_nans: If True, exclude NaN values from computation.
            run_stat: Callable (``NMStatWin._run_stat``) that executes a
                single stat dict and appends the result to the results list.
            bsln_result: Dict of baseline result keys (e.g., ``"s"``,
                ``"std"``), or ``{}`` if no baseline is used.
        """
        raise NotImplementedError(
            "%s.compute() not implemented" % self.__class__.__name__
        )


class NMStatFuncBasic(NMStatFunc):
    """Stat functions that take no extra parameters.

    Accepted names: median, mean, mean+var, mean+std, mean+sem, var, std,
    sem, rms, sum, pathlength, area, slope, value@x0, value@x1, count,
    count_nans, count_infs.

    Args:
        name: One of the names in ``FUNC_NAMES_BASIC``.
    """

    def __init__(self, name: str) -> None:
        if name.lower() not in FUNC_NAMES_BASIC:
            raise ValueError("func name: '%s'" % name)
        super().__init__(name.lower())

    def compute(self, data, x0, x1, xclip, ignore_nans, run_stat,
                bsln_result):
        """Run a single stat call and optionally record the baseline delta."""
        r = run_stat(data, {"name": self._name}, "main",
                     x0, x1, xclip, ignore_nans)
        self._add_ds(r, bsln_result)


class NMStatFuncMaxMin(NMStatFunc):
    """Max/min stat functions with optional averaging around the peak.

    Accepted names: max, min, mean@max, mean@min.

    When ``n_avg`` is provided for a ``"max"`` or ``"min"`` name, the name is
    upgraded to ``"mean@max"`` or ``"mean@min"`` automatically.

    Args:
        name: One of the names in ``FUNC_NAMES_MAXMIN``.
        n_avg: Number of points to average around the peak location. Required
            for ``mean@max`` / ``mean@min``; upgrades ``max`` / ``min`` when
            provided.
    """

    _VALID_KEYS = {"n_avg"}

    def __init__(
        self, name: str, n_avg: int | None = None,
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
        super().__init__(f)
        self._n_avg = n_avg

    def to_dict(self) -> dict:
        d = {"name": self._name}
        if self._n_avg is not None:
            d["n_avg"] = self._n_avg
        return d

    def compute(self, data, x0, x1, xclip, ignore_nans, run_stat,
                bsln_result):
        """Run the max/min stat; warn if n_avg <= 1 for mean@ variants."""
        func: dict[str, Any] = {"name": self._name}
        if self._n_avg is not None:
            func["n_avg"] = self._n_avg
        if self._name in ("mean@max", "mean@min"):
            n_avg = self._n_avg if self._n_avg is not None else 0
            if n_avg <= 1:
                func["warning"] = "not enough data points to compute a mean"
        r = run_stat(data, func, "main", x0, x1, xclip, ignore_nans)
        self._add_ds(r, bsln_result)


class NMStatFuncLevel(NMStatFunc):
    """Level crossing with an explicit absolute y-value threshold.

    Accepted names: level, level+, level-.

    Use ``NMStatFuncLevelNstd`` for a baseline-relative threshold.

    Args:
        name: One of the names in ``FUNC_NAMES_LEVEL``.
        ylevel: The absolute y threshold. Must be finite and not None.
    """

    _VALID_KEYS = {"ylevel"}

    def __init__(
        self, name: str, ylevel: float | None = None,
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
        super().__init__(name.lower())
        self._ylevel = ylevel

    def to_dict(self) -> dict:
        return {"name": self._name, "ylevel": self._ylevel}

    def compute(self, data, x0, x1, xclip, ignore_nans, run_stat,
                bsln_result):
        """Find the level crossing at ylevel and optionally add baseline delta."""
        func: dict[str, Any] = {"name": self._name, "ylevel": self._ylevel}
        r = run_stat(data, func, "main", x0, x1, xclip, ignore_nans)
        self._add_ds(r, bsln_result)


class NMStatFuncLevelNstd(NMStatFunc):
    """Level crossing at baseline mean +/- n_std standard deviations.

    Accepted names: level, level+, level-.

    Requires a ``"mean+std"`` baseline computation so that the threshold
    ``ylevel = bsln_mean + n_std * bsln_std`` can be derived at runtime.

    Args:
        name: One of the names in ``FUNC_NAMES_LEVEL``.
        n_std: Number of standard deviations from the baseline mean. Must
            be non-zero and finite.
    """

    _VALID_KEYS = {"n_std"}

    def __init__(
        self, name: str, n_std: float | None = None,
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
        super().__init__(name.lower())
        self._n_std = n_std

    def to_dict(self) -> dict:
        return {"name": self._name, "n_std": self._n_std}

    @property
    def needs_baseline(self) -> bool:
        return True

    def validate_baseline(self, bsln_func_name: str | None) -> None:
        """Raise RuntimeError if the baseline func is not ``"mean+std"``."""
        if bsln_func_name is None or bsln_func_name.lower() != "mean+std":
            raise RuntimeError(
                "level n_std requires baseline 'mean+std' computation"
            )

    def compute(self, data, x0, x1, xclip, ignore_nans, run_stat,
                bsln_result):
        """Derive ylevel from the baseline, then find the level crossing."""
        ylevel = math.nan
        if "s" in bsln_result and "std" in bsln_result:
            s = bsln_result["s"]
            std = bsln_result["std"]
            if not badvalue(s) and not badvalue(std):
                ylevel = s + self._n_std * std
        func: dict[str, Any] = {"name": self._name, "ylevel": ylevel}
        r = run_stat(data, func, "main", x0, x1, xclip, ignore_nans)
        self._add_ds(r, bsln_result)


class NMStatFuncRiseTime(NMStatFunc):
    """Rise time functions (risetime+/-, risetimeslope+/-).

    Measures the time for a signal to rise from ``p0``% to ``p1``% of its
    peak amplitude above baseline. Both percentages are required, and
    ``p0`` must be less than ``p1`` (e.g., p0=10, p1=90 for 10–90% rise
    time).

    Requires a mean or median baseline to establish the amplitude reference.

    Args:
        name: One of the names in ``FUNC_NAMES_RISETIME``.
        p0: Lower amplitude threshold as a percentage (0 < p0 < 100).
        p1: Upper amplitude threshold as a percentage (p0 < p1 < 100).
    """

    _VALID_KEYS = {"p0", "p1"}

    def __init__(
        self, name: str, p0: float | None = None,
        p1: float | None = None,
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
        super().__init__(f)
        self._p0 = p0
        self._p1 = p1

    def to_dict(self) -> dict:
        return {"name": self._name, "p0": self._p0, "p1": self._p1}

    @property
    def needs_baseline(self) -> bool:
        return True

    def validate_baseline(self, bsln_func_name: str | None) -> None:
        """Raise RuntimeError if baseline is not a mean or median func."""
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
        """Compute rise time: peak → level at p0% → level at p1%.

        Records ``dx = x(p1) - x(p0)`` and optionally the slope between them.
        """
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


class NMStatFuncFallTime(NMStatFunc):
    """Fall time functions (falltime+/-, falltimeslope+/-).

    Measures the time for a signal to fall from ``p0``% to ``p1``% of its
    peak amplitude above baseline, starting from the peak. Both percentages
    are required, and ``p0`` must be greater than ``p1`` (e.g., p0=90,
    p1=10 for 90–10% fall time).

    Requires a mean or median baseline to establish the amplitude reference.

    Args:
        name: One of the names in ``FUNC_NAMES_FALLTIME``.
        p0: Upper amplitude threshold as a percentage (0 < p1 < p0 < 100).
        p1: Lower amplitude threshold as a percentage (0 < p1 < p0 < 100).
        parent: Optional parent object reference.
    """

    _VALID_KEYS = {"p0", "p1"}

    def __init__(
        self, name: str, p0: float | None = None,
        p1: float | None = None,
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
        if p1 is None:
            raise KeyError("missing func key 'p1'")
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
        super().__init__(f)
        self._p0 = p0
        self._p1 = p1

    def to_dict(self) -> dict:
        return {"name": self._name, "p0": self._p0, "p1": self._p1}

    @property
    def needs_baseline(self) -> bool:
        return True

    def validate_baseline(self, bsln_func_name: str | None) -> None:
        """Raise RuntimeError if baseline is not a mean or median func."""
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
        """Compute fall time: peak → level at p0% → level at p1% after peak.

        Records ``dx = x(p1) - x(p0)`` and optionally the slope between them.
        """
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
            run_stat(data, {"name": "slope"}, f, r0["x"], r1["x"],
                     xclip, ignore_nans)


class NMStatFuncDecayTime(NMStatFunc):
    """Decay time functions (decaytime+/-).

    Measures the time from the signal peak to when it has decayed to p0%
    of its amplitude above baseline. Default p0 = 1/e ≈ 36.79%,
    giving one exponential time constant (tau).
    """

    _VALID_KEYS = {"p0"}

    def __init__(
        self, name: str, p0: float | None = None,
    ) -> None:
        if name.lower() not in FUNC_NAMES_DECAYTIME:
            raise ValueError("func name: '%s'" % name)
        f = name.lower()
        if p0 is None:
            p0 = DECAY_TIME_DEFAULT_PCT
        if isinstance(p0, bool):
            raise TypeError("p0: '%s'" % p0)
        p0 = float(p0)
        if math.isnan(p0) or math.isinf(p0):
            raise ValueError("p0: '%s'" % p0)
        if not (p0 > 0 and p0 < 100):
            raise ValueError("bad percent p0: %s" % p0)
        super().__init__(f)
        self._p0 = p0

    def to_dict(self) -> dict:
        return {"name": self._name, "p0": self._p0}

    @property
    def needs_baseline(self) -> bool:
        return True

    def validate_baseline(self, bsln_func_name: str | None) -> None:
        """Raise RuntimeError if baseline is not a mean or median func."""
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
        """Compute decay time: peak → level at p0% after peak.

        Records ``dx = x(p0) - peak_x``.
        """
        f = self._name

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
            r0["dx"] = math.nan
            return
        r0["dx"] = r0["x"] - peak_x


class NMStatFuncFWHM(NMStatFunc):
    """Full-width at half-maximum functions (fwhm+, fwhm-).

    Measures the width of a peak at ``p0``% of the amplitude before the peak
    and ``p1``% after, both defaulting to 50 (half maximum).
    ``dx = x(p1, after peak) - x(p0, before peak)``.

    Requires a mean or median baseline to establish the amplitude reference.

    Args:
        name: One of the names in ``FUNC_NAMES_FWHM``.
        p0: Threshold before the peak as a percentage (default 50).
        p1: Threshold after the peak as a percentage (default 50).
        parent: Optional parent object reference.
    """

    _VALID_KEYS = {"p0", "p1"}

    def __init__(
        self, name: str, p0: float | None = None,
        p1: float | None = None,
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
        super().__init__(f)
        self._p0 = p0
        self._p1 = p1

    def to_dict(self) -> dict:
        return {"name": self._name, "p0": self._p0, "p1": self._p1}

    @property
    def needs_baseline(self) -> bool:
        return True

    def validate_baseline(self, bsln_func_name: str | None) -> None:
        """Raise RuntimeError if baseline is not a mean or median func."""
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
        """Compute FWHM: peak → level at p0% before peak, level at p1% after.

        Records ``dx = x(p1) - x(p0)``.
        """
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


# =========================================================================
# Registry and factory
# =========================================================================

# Registry mapping func names to their class
_STAT_FUNC_REGISTRY: dict[str, type[NMStatFunc]] = {}
for _name in FUNC_NAMES_BASIC:
    _STAT_FUNC_REGISTRY[_name] = NMStatFuncBasic
for _name in FUNC_NAMES_MAXMIN:
    _STAT_FUNC_REGISTRY[_name] = NMStatFuncMaxMin
for _name in FUNC_NAMES_LEVEL:
    _STAT_FUNC_REGISTRY[_name] = NMStatFuncLevel
for _name in FUNC_NAMES_RISETIME:
    _STAT_FUNC_REGISTRY[_name] = NMStatFuncRiseTime
for _name in FUNC_NAMES_FALLTIME:
    _STAT_FUNC_REGISTRY[_name] = NMStatFuncFallTime
for _name in FUNC_NAMES_DECAYTIME:
    _STAT_FUNC_REGISTRY[_name] = NMStatFuncDecayTime
for _name in FUNC_NAMES_FWHM:
    _STAT_FUNC_REGISTRY[_name] = NMStatFuncFWHM


def _stat_func_from_dict(
    d: dict | str | None,
) -> NMStatFunc | None:
    """Create an NMStatFunc from a dict or string name.

    Args:
        d: Dict with at least a ``"name"`` key, a bare string func name,
            or None. Additional keys (``"p0"``, ``"n_avg"``, etc.) are
            forwarded to the subclass constructor.

    Returns:
        An NMStatFunc subclass instance, or None if ``d`` is None or ``{}``.

    Raises:
        TypeError: If ``d`` is not a dict or string.
        KeyError: If ``"name"`` is missing, a required parameter is absent,
            an unknown key is present, or both ``"ylevel"`` and ``"n_std"``
            are supplied together for a level func.
        ValueError: If the func name is not in ``_STAT_FUNC_REGISTRY``, or a
            parameter value is invalid (e.g., out-of-range percentage).
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
    if f not in _STAT_FUNC_REGISTRY:
        raise ValueError("func_name: %s" % name)
    cls = _STAT_FUNC_REGISTRY[f]
    # Level dispatch: redirect to NMStatFuncLevelNstd when "n_std" present
    if cls is NMStatFuncLevel:
        lower_keys = {k.lower() for k in d if k.lower() != "name"}
        if "n_std" in lower_keys and "ylevel" in lower_keys:
            raise KeyError("either 'ylevel' or 'n_std' is allowed, not both")
        if "n_std" in lower_keys:
            cls = NMStatFuncLevelNstd
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
    return cls(f, **kwargs)
