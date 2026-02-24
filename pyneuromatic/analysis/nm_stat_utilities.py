# -*- coding: utf-8 -*-
"""
Low-level stat dispatch handlers, stat() and stats() functions, and math
utilities (find_level_crossings, linear_regression).

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

import numpy as np

from pyneuromatic.core.nm_data import NMData
import pyneuromatic.core.nm_utilities as nmu


# =========================================================================
# stat() dispatch handlers (private)
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


_STAT_DISPATCH = {
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


# =========================================================================
# Public functions
# =========================================================================


def stat(
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

    results["data"] = data.path_str

    xunits = data.xscale.units
    yunits = data.yscale.units

    i0 = data.get_xindex(x0, clip=xclip)
    i1 = data.get_xindex(x1, clip=xclip)

    results["i0"] = i0
    results["i1"] = i1

    if i0 is None:
        e = "failed to compute i0 from x0"
        results["error"] = e
        return results
    if i1 is None:
        e = "failed to compute i1 from x1"
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

    handler = _STAT_DISPATCH.get(f)
    if handler is None:
        raise ValueError("unknown function '%s'" % func)

    return handler(**ctx)


def stats(
    yarray: np.ndarray,
    ignore_nans: bool = False,
    results: dict | None = None,
) -> dict:
    """Compute all summary statistics of a numpy array in a single pass.

    Computes mean, std, sem, rms, N, NaNs, INFs, min, max.

    Args:
        yarray: Input numpy array.
        ignore_nans: If True, exclude NaN values from calculations.
        results: Optional dict to populate. Created if None.

    Returns:
        Results dict with keys: mean, std, sem, rms, N, NaNs, INFs, min, max.
    """
    if not isinstance(yarray, np.ndarray):
        raise TypeError(nmu.type_error_str(yarray, "yarray", "NumPy ndarray"))
    if results is None:
        results = {}
    elif not isinstance(results, dict):
        raise TypeError(nmu.type_error_str(results, "results", "dictionary"))

    nans = int(np.count_nonzero(np.isnan(yarray)))
    infs = int(np.count_nonzero(np.isinf(yarray)))
    results["NaNs"] = nans
    results["INFs"] = infs

    if ignore_nans:
        arr = yarray[~np.isnan(yarray)]
    else:
        arr = yarray

    n = arr.size
    results["N"] = n

    if n == 0:
        results["mean"] = math.nan
        results["std"] = math.nan
        results["sem"] = math.nan
        results["rms"] = math.nan
        results["min"] = math.nan
        results["max"] = math.nan
        return results

    results["mean"] = float(np.mean(arr))
    results["std"] = float(np.std(arr, ddof=1) if n > 1 else math.nan)
    results["sem"] = (results["std"] / math.sqrt(n)
                      if n > 1 else math.nan)
    results["rms"] = float(math.sqrt(np.mean(np.square(arr))))
    results["min"] = float(np.min(arr))
    results["max"] = float(np.max(arr))

    return results


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
    locations = np.argwhere(level_crossings)  # grab True locations
    if len(locations.shape) != 2:  # shape = (N, 1)
        raise RuntimeError("locations shape should be 2")
    locations = locations[:, 0]
    indexes = []
    xvalues = []

    for i in locations:

        if i == 0:
            continue

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
