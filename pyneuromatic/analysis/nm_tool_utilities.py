# -*- coding: utf-8 -*-
"""
NMData-aware wrappers for nm_math functions.

Convenience functions that accept NMData objects and automatically unpack
x-scale parameters (start, delta, units) before delegating to the
corresponding pure-numpy functions in nm_math.py.

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

from pyneuromatic.core.nm_data import NMData
import pyneuromatic.core.nm_math as nm_math


def _sample_rate_from_xscale(data: NMData) -> float:
    """Derive sample rate (Hz) from *data*.xscale.delta and xscale.units.

    Args:
        data: NMData whose xscale.delta gives the sample interval.

    Returns:
        Sample rate in Hz (samples per second).

    Raises:
        ValueError: If data has a non-uniform xarray (variable sample rate),
            or if xscale.delta is zero or unset.
    """
    if data.xarray is not None:
        raise ValueError(
            "cannot derive a single sample_rate: data has a non-uniform "
            "xarray (variable sample rate); filtering is not supported"
        )
    delta = data.xscale.delta
    units = data.xscale.units
    if not delta or delta == 0:
        raise ValueError(
            "cannot derive sample_rate: xscale.delta is zero or unset"
        )
    factor = nm_math.si_scale_factor(units, "s") if units else 1.0
    return 1.0 / (delta * factor)


def find_level_crossings_nmdata(
    data: NMData,
    ylevel: float,
    func_name: str = "level",
    x0: float = -math.inf,
    x1: float = math.inf,
    x_interp: bool = True,
    ignore_nans: bool = True,
) -> tuple:
    """Find threshold crossings in an NMData array.

    Convenience wrapper around :func:`~pyneuromatic.core.nm_math.find_level_crossings`
    that unpacks x-scale parameters from *data* automatically.  If *data* has
    a non-uniform ``xarray`` (e.g. variable sample rate), it is passed through
    directly; otherwise ``xscale.start`` and ``xscale.delta`` are used.

    Args:
        data:        NMData containing the y-values and x-scale.
        ylevel:      Y-axis threshold.
        func_name:   Crossing direction: ``"level"`` (all), ``"level+"``
                     (rising), or ``"level-"`` (falling).
        x0:          Window start. Default ``-inf`` (no lower bound).
                     If *x0* > *x1*, crossings are returned in descending
                     x order (backwards search).
        x1:          Window end. Default ``+inf`` (no upper bound).
        x_interp:    If True (default), return interpolated x at crossing.
        ignore_nans: If True (default), ignore NaN values.

    Returns:
        Tuple ``(indexes, xvalues)`` of numpy arrays.
    """
    if data.xarray is not None:
        return nm_math.find_level_crossings(
            data.nparray,
            ylevel,
            func_name=func_name,
            xarray=data.xarray,
            x0=x0,
            x1=x1,
            x_interp=x_interp,
            ignore_nans=ignore_nans,
        )
    return nm_math.find_level_crossings(
        data.nparray,
        ylevel,
        func_name=func_name,
        xstart=data.xscale.start,
        xdelta=data.xscale.delta,
        x0=x0,
        x1=x1,
        x_interp=x_interp,
        ignore_nans=ignore_nans,
    )


def linear_regression_nmdata(
    data: NMData,
    ignore_nans: bool = True,
) -> tuple:
    """Fit a line to an NMData array.

    Convenience wrapper around :func:`~pyneuromatic.core.nm_math.linear_regression`
    that unpacks x-scale parameters from *data* automatically.  If *data* has
    a non-uniform ``xarray``, it is passed through directly; otherwise
    ``xscale.start`` and ``xscale.delta`` are used.

    Args:
        data:        NMData containing the y-values and x-scale.
        ignore_nans: If True (default), ignore NaN values.

    Returns:
        Tuple ``(slope, intercept)``.
    """
    if data.xarray is not None:
        return nm_math.linear_regression(
            data.nparray,
            xarray=data.xarray,
            ignore_nans=ignore_nans,
        )
    return nm_math.linear_regression(
        data.nparray,
        xstart=data.xscale.start,
        xdelta=data.xscale.delta,
        ignore_nans=ignore_nans,
    )


def filter_butterworth_nmdata(
    data: NMData,
    cutoff: float | list[float],
    order: int = 4,
    btype: str = "low",
) -> np.ndarray:
    """Apply a Butterworth filter to an NMData array.

    Convenience wrapper around :func:`~pyneuromatic.core.nm_math.filter_butterworth`
    that derives the sample rate from *data*.xscale automatically.

    Args:
        data:   NMData containing the signal and x-scale.
        cutoff: Cut-off frequency in the same units as the sample rate (Hz).
        order:  Filter order. Default 4.
        btype:  Filter type: ``"low"``, ``"high"``, or ``"band"``.

    Returns:
        Filtered numpy array (same shape as input).
    """
    sr = _sample_rate_from_xscale(data)
    return nm_math.filter_butterworth(data.nparray, cutoff, sr, order, btype)


def filter_bessel_nmdata(
    data: NMData,
    cutoff: float | list[float],
    order: int = 4,
    btype: str = "low",
) -> np.ndarray:
    """Apply a Bessel filter to an NMData array.

    Convenience wrapper around :func:`~pyneuromatic.core.nm_math.filter_bessel`
    that derives the sample rate from *data*.xscale automatically.

    Args:
        data:   NMData containing the signal and x-scale.
        cutoff: Cut-off frequency in Hz.
        order:  Filter order. Default 4.
        btype:  Filter type: ``"low"``, ``"high"``, or ``"band"``.

    Returns:
        Filtered numpy array (same shape as input).
    """
    sr = _sample_rate_from_xscale(data)
    return nm_math.filter_bessel(data.nparray, cutoff, sr, order, btype)


def filter_notch_nmdata(
    data: NMData,
    freq: float,
    q: float = 30.0,
) -> np.ndarray:
    """Apply a notch filter to an NMData array.

    Convenience wrapper around :func:`~pyneuromatic.core.nm_math.filter_notch`
    that derives the sample rate from *data*.xscale automatically.

    Args:
        data: NMData containing the signal and x-scale.
        freq: Notch frequency in Hz.
        q:    Quality factor. Default 30.0.

    Returns:
        Filtered numpy array (same shape as input).
    """
    sr = _sample_rate_from_xscale(data)
    return nm_math.filter_notch(data.nparray, freq, sr, q)
