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
    xbgn: float = -math.inf,
    xend: float = math.inf,
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
        xbgn:          Window start. Default ``-inf`` (no lower bound).
                     If *xbgn* > *xend*, crossings are returned in descending
                     x order (backwards search).
        xend:          Window end. Default ``+inf`` (no upper bound).
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
            xbgn=xbgn,
            xend=xend,
            x_interp=x_interp,
            ignore_nans=ignore_nans,
        )
    return nm_math.find_level_crossings(
        data.nparray,
        ylevel,
        func_name=func_name,
        xstart=data.xscale.start,
        xdelta=data.xscale.delta,
        xbgn=xbgn,
        xend=xend,
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


def find_events_nmdata(
    data: NMData,
    algorithm: str,
    polarity: str,
    value: float = 20.0,
    baseline_avg: float = 2.0,
    baseline_dt: float = 2.0,
    template: np.ndarray | None = None,
    criterion_threshold: float = 4.0,
    template_baseline: float = 0.0,
    refractory: float = 0.0,
    xbgn: float = -math.inf,
    xend: float = math.inf,
    onset_search: bool = False,
    onset_avg: float = 1.0,
    onset_nstdv: float = 1.0,
    onset_limit: float = 5.0,
    peak_search: bool = False,
    peak_avg: float = 1.0,
    peak_nstdv: float = 1.0,
    peak_limit: float = 10.0,
    max_events: int = 0,
    match_criterion: np.ndarray | None = None,
) -> dict:
    """Find spontaneous events in an NMData array.

    Convenience wrapper around the nm_math event-detection functions that
    unpacks x-scale parameters from *data* automatically. Supports three
    detection algorithms, optional onset refinement, and optional peak
    refinement (Kudoh & Taguchi 2002; Clements & Bekkers 1997).

    Args:
        data: NMData containing the y-values and x-scale.
        algorithm: Detection algorithm — ``"threshold"``, ``"nstdv"``,
            or ``"template"``.
        polarity: ``"negative"`` or ``"positive"``.
        value: Threshold magnitude (threshold/nstdv algorithms).
            Ignored for template algorithm.
        baseline_avg: Baseline averaging window (x-units, threshold/nstdv).
        baseline_dt: Detection window after t0 (x-units, threshold/nstdv).
        template: 1-D numpy array (template algorithm only). Normalized to
            [0, 1] internally before calling match_template. If a baseline
            window is desired, it must already be included in this array
            (e.g. leading zeros prepended by the caller). Used only when
            *match_criterion* is ``None``.
        criterion_threshold: Criterion threshold for template matching (default 4).
        template_baseline: Offset (x-units) applied to shift detected times
            forward after criterion level-crossing detection. Use this to
            correct for a baseline window included at the start of the
            template: a baseline of length *B* causes the criterion crossing
            to occur *B* before the true event onset, and this shift recovers
            that offset. Default 0.0 (no shift). Note: the baseline prepending
            itself is the caller's responsibility (see above).
        refractory: Minimum inter-event interval (x-units, all algorithms).
        xbgn: Search start (x-units). Default ``-inf``.
        xend: Search end (x-units). Default ``+inf``.
        onset_search: If True, search backward from each t_event for onset.
        onset_avg: Onset window size (x-units).
        onset_nstdv: Onset N×stdv.
        onset_limit: Max backward search distance (x-units).
        peak_search: If True, search forward from each t_event for peak.
        peak_avg: Peak window size (x-units).
        peak_nstdv: Peak N×stdv.
        peak_limit: Max forward search distance (x-units).
        max_events: Stop after this many accepted events. 0 means no limit
            (default).
        match_criterion: Pre-computed template criterion array (template algorithm
            only). When provided, ``match_template()`` is skipped. Pass the
            cached result of a prior call to avoid recomputing for long
            recordings.

    Returns:
        Dict with keys:
        ``"detect_times"`` — list[float], accepted event x-times.
        ``"onset_times"``  — list[float | None], onset per accepted event.
        ``"peak_times"``   — list[float | None], peak per accepted event.
        ``"reject_times"`` — list[float], t_event for rejected events.
        ``"match_criterion"`` — np.ndarray | None, criterion wave (template
            algorithm only, else None).
        ``"xunits"``       — str, from data.xscale.units.
    """
    yarray = data.nparray
    xstart = data.xscale.start if data.xscale.start is not None else 0.0
    xdelta = data.xscale.delta if data.xscale.delta is not None else 1.0
    xunits = data.xscale.units or ""

    result: dict = {
        "detect_times":    [],
        "onset_times":     [],
        "peak_times":      [],
        "reject_times":    [],
        "match_criterion": None,
        "xunits":          xunits,
    }

    if yarray is None or len(yarray) == 0:
        return result

    if algorithm == "template":
        if match_criterion is None:
            if template is None:
                raise ValueError("template must be provided when algorithm='template'")
            tpl = template.astype(float, copy=True)
            tpl_min = tpl.min()
            tpl_max = tpl.max()
            if tpl_max != tpl_min:
                tpl = (tpl - tpl_min) / (tpl_max - tpl_min)
            match_criterion = nm_math.match_template(yarray, tpl)
        result["match_criterion"] = match_criterion
        func_name = "level-" if polarity == "negative" else "level+"
        thresh = -criterion_threshold if polarity == "negative" else criterion_threshold
        _idxs, candidate_times = nm_math.find_level_crossings(
            match_criterion,
            thresh,
            func_name=func_name,
            xstart=xstart,
            xdelta=xdelta,
            xbgn=xbgn,
            xend=xend,
        )
        candidate_times = list(candidate_times)
        # Shift times to recover true event onset (baseline offset correction)
        if template_baseline > 0:
            candidate_times = [t + template_baseline for t in candidate_times]
        # Apply refractory filter
        if refractory > 0 and len(candidate_times) > 1:
            filtered = [candidate_times[0]]
            for t in candidate_times[1:]:
                if t - filtered[-1] >= refractory:
                    filtered.append(t)
            candidate_times = filtered
    else:
        mode = "nstdv" if algorithm == "nstdv" else "threshold"
        candidate_times = nm_math.find_events_sliding_baseline(
            yarray,
            xstart=xstart,
            xdelta=xdelta,
            polarity=polarity,
            mode=mode,
            threshold=value,
            baseline_avg=baseline_avg,
            baseline_dt=baseline_dt,
            refractory=refractory,
            xbgn=xbgn,
            xend=xend,
            max_events=max_events,
        )

    for t_event in candidate_times:
        t_onset = None
        t_peak  = None
        rejected = False

        if onset_search:
            t_onset = nm_math.find_event_onset(
                yarray, xstart, xdelta, t_event,
                polarity=polarity,
                avg=onset_avg,
                nstdv=onset_nstdv,
                limit=onset_limit,
            )
            if t_onset is None:
                rejected = True

        if not rejected and peak_search:
            t_peak = nm_math.find_event_peak(
                yarray, xstart, xdelta, t_event,
                polarity=polarity,
                avg=peak_avg,
                nstdv=peak_nstdv,
                limit=peak_limit,
            )
            if t_peak is None:
                rejected = True

        if rejected:
            result["reject_times"].append(t_event)
        else:
            result["detect_times"].append(t_event)
            result["onset_times"].append(t_onset)
            result["peak_times"].append(t_peak)
            if max_events > 0 and len(result["detect_times"]) >= max_events:
                break

    return result


def fit_nmdata(
    data: NMData,
    func_name: str,
    xbgn: float = -math.inf,
    xend: float = math.inf,
    degree: int = 2,
    x_origin: float = 0.0,
    p0: dict | None = None,
    sigma: np.ndarray | None = None,
    maxfev: int = 10000,
    ignore_nans: bool = True,
) -> dict:
    """Fit a curve to an NMData array.

    Convenience wrapper around :func:`~pyneuromatic.core.nm_math.fit_line`,
    :func:`~pyneuromatic.core.nm_math.fit_poly`,
    :func:`~pyneuromatic.core.nm_math.fit_exp`, and
    :func:`~pyneuromatic.core.nm_math.fit_gauss` that unpacks x-scale
    parameters from *data* automatically.

    Args:
        data:        NMData containing the y-values and x-scale.
        func_name:   Fitting function — ``"line"``, ``"poly"``, ``"exp"``,
                     or ``"gauss"``.
        xbgn:          Fit window start (x-units). Default ``-inf``.
        xend:          Fit window end (x-units). Default ``+inf``.
        degree:      Polynomial degree (``func_name="poly"`` only). Default 2.
        p0:          Initial parameter estimates for nonlinear fits
                     (``func_name="exp"`` or ``"gauss"`` only). Dict with
                     optional keys specific to each model.
        sigma:       Per-point standard deviations for weighted fitting.
        ignore_nans: If True (default), exclude NaN values before fitting.

    Returns:
        Result dict from the underlying ``nm_math.fit_*`` function.

    Raises:
        ValueError: If *func_name* is not one of the supported values.
    """
    xstart = data.xscale.start if data.xscale.start is not None else 0.0
    xdelta = data.xscale.delta if data.xscale.delta is not None else 1.0
    xarray = data.xarray

    common = dict(xbgn=xbgn, xend=xend, ignore_nans=ignore_nans)
    if xarray is not None:
        common["xarray"] = xarray
    else:
        common["xstart"] = xstart
        common["xdelta"] = xdelta

    if func_name == "line":
        return nm_math.fit_line(data.nparray, sigma=sigma, **common)
    if func_name.startswith("poly"):
        return nm_math.fit_poly(data.nparray, degree=degree, sigma=sigma, **common)
    if func_name == "exp":
        return nm_math.fit_exp(data.nparray, x_origin=x_origin, p0=p0, sigma=sigma, maxfev=maxfev, **common)
    if func_name == "exp2":
        return nm_math.fit_exp2(data.nparray, x_origin=x_origin, p0=p0, sigma=sigma, maxfev=maxfev, **common)
    if func_name == "gauss":
        return nm_math.fit_gauss(data.nparray, p0=p0, sigma=sigma, maxfev=maxfev, **common)
    if func_name == "boltzmann":
        return nm_math.fit_boltzmann(data.nparray, p0=p0, sigma=sigma, maxfev=maxfev, **common)
    raise ValueError(
        "fit_nmdata: func_name must be 'line', 'poly2'–'poly9', 'exp', 'exp2', "
        "'gauss', or 'boltzmann', got %r" % func_name
    )


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
