"""Pure math utility functions shared across pyNeuroMatic.

No NM object dependencies — only numpy/math and nm_utilities.
"""
from __future__ import annotations

import math

import numpy as np

import pyneuromatic.core.nm_utilities as nmu

# =========================================================================
# Inequality helpers
# =========================================================================

_SINGLE_INEQUALITY_OPS: frozenset[str] = frozenset({">", ">=", "<", "<=", "==", "!="})
_RANGE_INEQUALITY_OPS: frozenset[str] = frozenset({"<<", "<=<=", "<=<", "<<="})
VALID_INEQUALITY_OPS: frozenset[str] = _SINGLE_INEQUALITY_OPS | _RANGE_INEQUALITY_OPS


def inequality_mask(
    arr: np.ndarray,
    op: str,
    a: float,
    b: float | None = None,
) -> np.ndarray:
    """Return a boolean mask: True where *arr* satisfies the inequality.

    NaN elements propagate as False (numpy comparison behaviour).

    Args:
        arr: 1-D (or N-D) numpy array of values to test.
        op:  Comparison operator.  Single-threshold: ``">"``, ``">="``,
             ``"<"``, ``"<="``, ``"=="``, ``"!="``.  Range (requires *b*):
             ``"<<"`` (a < y < b), ``"<=<="`` (a <= y <= b),
             ``"<=<"`` (a <= y < b), ``"<<="`` (a < y <= b).
        a:   Threshold value (or lower bound for range operators).
        b:   Upper bound; required for range operators.

    Returns:
        Boolean numpy array, same shape as *arr*.

    Raises:
        ValueError: If *op* is not in ``VALID_INEQUALITY_OPS``, or a range
            op is used without providing *b*.
    """
    if op not in VALID_INEQUALITY_OPS:
        raise ValueError(
            "unknown operator %r. Valid ops: %s" % (op, sorted(VALID_INEQUALITY_OPS))
        )
    if op in _RANGE_INEQUALITY_OPS and b is None:
        raise ValueError("range operator %r requires b to be specified" % op)

    if op == ">":
        return arr > a
    if op == ">=":
        return arr >= a
    if op == "<":
        return arr < a
    if op == "<=":
        return arr <= a
    if op == "==":
        return arr == a
    if op == "!=":
        return arr != a
    if op == "<<":      # a < y < b
        return (arr > a) & (arr < b)
    if op == "<=<=":    # a <= y <= b
        return (arr >= a) & (arr <= b)
    if op == "<=<":     # a <= y < b
        return (arr >= a) & (arr < b)
    # op == "<<="       # a < y <= b
    return (arr > a) & (arr <= b)


def apply_inequality(
    arr: np.ndarray,
    op: str,
    a: float,
    b: float | None = None,
    binary_output: bool = True,
) -> np.ndarray:
    """Apply an inequality test and return the result array.

    Args:
        arr:           Array to test.
        op:            Comparison operator (see :func:`inequality_mask`).
        a:             Threshold / lower bound.
        b:             Upper bound; required for range ops.
        binary_output: If True (default), returns 1.0 where mask is True,
                       0.0 elsewhere.  If False, returns the original value
                       where mask is True and NaN elsewhere.

    Returns:
        Float numpy array, same shape as *arr*.
    """
    mask = inequality_mask(arr, op, a, b)
    if binary_output:
        return mask.astype(float)
    return np.where(mask, arr, np.nan)


def inequality_condition_str(op: str, a: float, b: float | None) -> str:
    """Build a human-readable condition string.

    Examples: ``"y > 50"``, ``"2 < y < 5"``, ``"y == 0"``.
    """
    if op == ">":    return "y > %g" % a
    if op == ">=":   return "y >= %g" % a
    if op == "<":    return "y < %g" % a
    if op == "<=":   return "y <= %g" % a
    if op == "==":   return "y == %g" % a
    if op == "!=":   return "y != %g" % a
    if op == "<<":   return "%g < y < %g" % (a, b)
    if op == "<=<=": return "%g <= y <= %g" % (a, b)
    if op == "<=<":  return "%g <= y < %g" % (a, b)
    if op == "<<=":  return "%g < y <= %g" % (a, b)
    return ""


# =========================================================================
# Arithmetic helpers
# =========================================================================

VALID_ARITH_OPS: frozenset[str] = frozenset({"x", "/", "+", "-", "=", "**"})


def apply_arithmetic(arr: np.ndarray, value, op: str) -> np.ndarray:
    """Apply a binary arithmetic operation between *arr* and *value*.

    Args:
        arr:   Input array.
        value: Scalar float or ndarray of the same length as *arr*.
        op:    One of ``"x"`` (multiply), ``"/"`` (divide), ``"+"`` (add),
               ``"-"`` (subtract), ``"="`` (assign constant/array),
               ``"**"`` (exponentiate).

    Returns:
        Result array (same dtype promotion as numpy default, except ``"="``
        which returns float64).

    Raises:
        ValueError: If *op* is not in ``VALID_ARITH_OPS``.
    """
    if op == "x":
        return arr * value
    if op == "/":
        return arr / value
    if op == "+":
        return arr + value
    if op == "-":
        return arr - value
    if op == "=":
        return np.full_like(arr, value, dtype=float)
    if op == "**":
        return arr ** value
    raise ValueError("unknown op: %r" % op)


# =========================================================================
# Reference-value helper
# =========================================================================


def compute_ref_value(arr: np.ndarray, fxn: str, n_mean: int) -> float:
    """Compute a scalar reference value from *arr* using the given function.

    Args:
        arr:    Array slice to compute the reference from.
        fxn:    One of ``"mean"``, ``"min"``, ``"max"``, ``"mean@min"``,
                ``"mean@max"``.
        n_mean: Number of points to average around the extremum (used
                only for ``"mean@min"`` and ``"mean@max"``).

    Returns:
        Scalar reference value, or ``float("nan")`` if *arr* is empty.
    """
    if len(arr) == 0:
        return float("nan")
    if fxn == "mean":
        return float(np.nanmean(arr))
    if fxn == "min":
        return float(np.nanmin(arr))
    if fxn == "max":
        return float(np.nanmax(arr))
    if fxn == "mean@min":
        i = int(np.nanargmin(arr))
        half = n_mean // 2
        return float(np.nanmean(arr[max(0, i - half):i + half + 1]))
    if fxn == "mean@max":
        i = int(np.nanargmax(arr))
        half = n_mean // 2
        return float(np.nanmean(arr[max(0, i - half):i + half + 1]))
    return float("nan")


# =========================================================================
# Xscale-window helper
# =========================================================================


def xscale_window_to_slice(
    arr: np.ndarray,
    xscale_dict: dict,
    xbgn: float,
    xend: float,
) -> slice:
    """Convert a xscale window to an array slice using xscale start/delta.

    Clips to valid range; returns an empty slice if the window is fully
    outside the array bounds.  Infinite bounds are treated as array
    boundaries (``-inf`` → index 0, ``+inf`` → index len(arr)).

    Args:
        arr:         Array whose length defines the valid index range.
        xscale_dict: Dict with ``"start"`` and ``"delta"`` keys (floats).
        xbgn:     Start of the xscale window.  ``-inf`` selects from the
                     beginning of the array.
        xend:       End of the xscale window (inclusive).  ``+inf`` selects
                     to the end of the array.

    Returns:
        A :class:`slice` suitable for indexing *arr*.
    """
    start = xscale_dict.get("start", 0.0)
    delta = xscale_dict.get("delta", 1.0)
    if delta == 0:
        return slice(0, 0)
    i0 = 0 if math.isinf(xbgn) else int(round((xbgn - start) / delta))
    i1 = len(arr) if math.isinf(xend) else int(round((xend - start) / delta)) + 1
    i0 = max(0, i0)
    i1 = min(len(arr), i1)
    return slice(i0, i1)


# =========================================================================
# Fluorescence dF/F₀ helper
# =========================================================================


def apply_dfof(arr: np.ndarray, f0: float) -> np.ndarray:
    """Return dF/F₀ = (arr − f0) / f0.

    If *f0* is zero, returns an array of NaN (avoids division by zero).

    Args:
        arr: Input fluorescence array (float).
        f0:  Baseline scalar F₀.

    Returns:
        ndarray with same shape as *arr*.
    """
    if f0 == 0.0:
        return np.full_like(arr, np.nan, dtype=float)
    return (arr - f0) / f0


# =========================================================================
# SI unit helpers
# =========================================================================

# SI prefix → base-10 exponent
_SI_PREFIX_EXPONENTS: dict[str, int] = {
    "f": -15,       # femto
    "p": -12,       # pico
    "n": -9,        # nano
    "u": -6,        # micro (ASCII)
    "\u00b5": -6,   # µ (U+00B5 micro sign)
    "\u03bc": -6,   # μ (U+03BC Greek mu)
    "m": -3,        # milli
    "":   0,        # (no prefix — base unit)
    "k":  3,        # kilo
    "M":  6,        # mega
    "G":  9,        # giga
    "T": 12,        # tera
}


def parse_si_units(units: str) -> tuple[str, str]:
    """Split an SI-prefixed units string into ``(prefix, base_unit)``.

    The first character is treated as a prefix if it is a known SI prefix
    AND the string is longer than one character.

    Examples::

        parse_si_units("pA")  → ("p",  "A")
        parse_si_units("mV")  → ("m",  "V")
        parse_si_units("V")   → ("",   "V")
        parse_si_units("ms")  → ("m",  "s")
        parse_si_units("kHz") → ("k",  "Hz")

    Args:
        units: Non-empty units string (e.g. ``"pA"``, ``"mV"``).

    Returns:
        Tuple ``(prefix, base_unit)`` where *prefix* is ``""`` when no
        recognised SI prefix is present.

    Raises:
        ValueError: If *units* is empty.
    """
    if not units:
        raise ValueError("units string is empty")
    first = units[0]
    if len(units) > 1 and first in _SI_PREFIX_EXPONENTS:
        return (first, units[1:])
    return ("", units)


def si_scale_factor(from_units: str, to_units: str) -> float:
    """Return the multiplicative factor to convert *from_units* → *to_units*.

    Both strings must share the same base unit (e.g. ``"A"``).

    Examples::

        si_scale_factor("pA", "nA") → 1e-3   (1 pA = 0.001 nA)
        si_scale_factor("mV", "V")  → 1e-3
        si_scale_factor("V",  "mV") → 1e3

    Supported base units include ``"V"`` (volts), ``"A"`` (amperes),
    ``"Ohm"`` or ``"Ω"`` (ohms), and ``"s"`` (seconds).

    Args:
        from_units: Source units string (e.g. ``"pA"``).
        to_units:   Target units string (e.g. ``"nA"``).

    Returns:
        Float scale factor.

    Raises:
        ValueError: If base units differ, or either prefix is unrecognised.
    """
    from_prefix, from_base = parse_si_units(from_units)
    to_prefix, to_base = parse_si_units(to_units)
    if from_base != to_base:
        raise ValueError(
            "base units must match: %r (from %r) vs %r (from %r)"
            % (from_base, from_units, to_base, to_units)
        )
    exp_diff = _SI_PREFIX_EXPONENTS[from_prefix] - _SI_PREFIX_EXPONENTS[to_prefix]
    return 10.0 ** exp_diff


# =========================================================================
# Array statistics
# =========================================================================


def array_stats(
    yarray: np.ndarray,
    ignore_nans: bool = False,
    results: dict | None = None,
) -> dict:
    """Compute all summary statistics of a numpy array in a single pass.

    Computes mean, std, sem, rms, N, NaNs, INFs, min, max.

    Args:
        yarray:      Input numpy array.
        ignore_nans: If True, exclude NaN values from calculations.
        results:     Optional dict to populate. Created if None.

    Returns:
        Results dict with keys: mean, std, sem, rms, N, NaNs, INFs, min, max.

    Raises:
        TypeError: If *yarray* is not a numpy ndarray or *results* is not
            a dict.
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
    results["sem"] = (results["std"] / math.sqrt(n) if n > 1 else math.nan)
    results["rms"] = float(math.sqrt(np.mean(np.square(arr))))
    results["min"] = float(np.min(arr))
    results["max"] = float(np.max(arr))

    return results


# =========================================================================
# Level crossings
# =========================================================================


def interp_x(ylevel: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """Interpolate the x-coordinate where a line segment crosses *ylevel*.

    Fits a line through (x0, y0) and (x1, y1) and returns the xvalue
    where that line equals *ylevel*.

    Args:
        ylevel: The y-axis threshold to intersect.
        x0, y0: Coordinates of the first point.
        x1, y1: Coordinates of the second point.

    Returns:
        Interpolated xvalue (float) at y = ylevel.
    """
    dx = x1 - x0
    dy = y1 - y0
    m = dy / dx
    b = y1 - m * x1
    x = (ylevel - b) / m
    return x


def find_level_crossings(
    yarray: np.ndarray,
    ylevel: float,
    func_name: str = "level",
    xarray: np.ndarray | None = None,
    xstart: float = 0,
    xdelta: float = 1,
    xbgn: float = -math.inf,
    xend: float = math.inf,
    x_interp: bool = True,
    ignore_nans: bool = True,
) -> tuple:
    """Find crossings of a y-axis level in a data array.

    A crossing is detected wherever the signal transitions across *ylevel*
    (i.e. ``np.diff(yarray > ylevel)`` is True). For each crossing, the
    nearest sample index and interpolated xvalue are returned.

    Args:
        yarray:      1-D numpy array of yvalues to search.
        ylevel:      The y-axis threshold to search for.
        func_name:   Controls which crossing directions are returned:
                     ``"level"`` — all crossings (both rising and falling).
                     ``"level+"`` — rising crossings only.
                     ``"level-"`` — falling crossings only.
        xarray:      Optional 1-D numpy array of xvalues (same size as
                     *yarray*). Used instead of *xstart*/*xdelta* when
                     provided.
        xstart:      X-scale start value; used when *xarray* is None.
        xdelta:      X-scale sample interval; used when *xarray* is None.
        xbgn:          X-axis window start. Only crossings at x >= xbgn are
                     returned. Default ``-inf`` (no lower bound).
                     If *xbgn* > *xend*, a backwards search is performed: the
                     window is ``[xend, xbgn]`` and crossings are returned in
                     descending x order.
        xend:          X-axis window end. Only crossings at x <= xend are
                     returned. Default ``+inf`` (no upper bound).
        x_interp:    If True (default), returns the interpolated xvalue at
                     the exact crossing. If False, returns the xvalue at
                     the nearest sample index.
        ignore_nans: If True (default), NaN values are skipped during
                     transition detection. A crossing between two non-NaN
                     samples is still detected even if NaN values lie
                     between them; the two bounding non-NaN samples and
                     their x positions are used to linearly interpolate
                     the crossing x location (Igor Pro behaviour).
                     If False, a NaN sample acts as ``False`` in the
                     ``y > ylevel`` comparison, which silently blocks
                     detection of crossings that span a NaN gap.

    Returns:
        Tuple ``(indexes, xvalues)`` of numpy arrays:
        *indexes* — sample indices (int) nearest to each crossing.
        *xvalues* — xvalues (float) at each crossing location.
        Both arrays are empty if no crossings are found.
        If *xbgn* > *xend* (backwards search), both arrays are in descending
        x order.

    Raises:
        TypeError:  If *func_name* is not a string, *yarray*/*xarray* is
                    not a numpy ndarray, or *xbgn*/*xend* is a bool.
        ValueError: If *func_name* does not contain ``"level"``, *ylevel*
                    is inf/nan, *xstart*/*xdelta* is inf/nan, *xbgn*/*xend*
                    is NaN, or *xarray* size differs from *yarray*.
    """
    if not isinstance(func_name, str):
        raise TypeError(nmu.type_error_str(func_name, "func_name", "string"))

    f = func_name.lower()
    if "level" not in f:
        raise ValueError("func_name: '%s'" % func_name)

    if isinstance(ylevel, float):
        if math.isinf(ylevel) or math.isnan(ylevel):
            raise ValueError("ylevel: '%s'" % ylevel)
    else:
        ylevel = float(ylevel)

    if not isinstance(yarray, np.ndarray):
        raise TypeError(nmu.type_error_str(yarray, "yarray", "NumPy.ndarray"))

    found_xarray = False

    if xarray is None:
        pass
    elif isinstance(xarray, np.ndarray):
        if xarray.size == yarray.size:
            found_xarray = True
        else:
            raise ValueError(
                "x-y paired NumPy arrays have different size: %s != %s"
                % (xarray.size, yarray.size)
            )
    else:
        raise TypeError(nmu.type_error_str(xarray, "xarray", "NumPy.ndarray"))

    if isinstance(xstart, float):
        if math.isinf(xstart) or math.isnan(xstart):
            raise ValueError("xstart: '%s'" % xstart)
    else:
        xstart = float(xstart)

    if isinstance(xdelta, float):
        if math.isinf(xdelta) or math.isnan(xdelta):
            raise ValueError("xdelta: '%s'" % xdelta)
    else:
        xdelta = float(xdelta)

    if not isinstance(x_interp, bool):
        x_interp = True
    if not isinstance(ignore_nans, bool):
        ignore_nans = True

    # Validate xbgn / xend window parameters
    if isinstance(xbgn, bool):
        raise TypeError(nmu.type_error_str(xbgn, "xbgn", "float"))
    if isinstance(xend, bool):
        raise TypeError(nmu.type_error_str(xend, "xend", "float"))
    xbgn = float(xbgn)
    xend = float(xend)
    if math.isnan(xbgn):
        raise ValueError("xbgn: '%s'" % xbgn)
    if math.isnan(xend):
        raise ValueError("xend: '%s'" % xend)

    backward = xbgn > xend
    x_low  = min(xbgn, xend)
    x_high = max(xbgn, xend)

    # NaN compaction (Igor Pro behaviour): remove NaN samples but keep their
    # x positions so that a crossing between two non-NaN samples separated by
    # a NaN gap is still detected via linear interpolation of their x values.
    orig_indices = None  # maps compact index → original yarray index
    if ignore_nans and np.any(np.isnan(yarray)):
        keep = ~np.isnan(yarray)
        orig_indices = np.where(keep)[0]
        # Build explicit x values for the kept samples before discarding yarray
        if found_xarray:
            xarray = xarray[keep]
        else:
            xarray = xstart + orig_indices * xdelta
            found_xarray = True
        yarray = yarray[keep]

    n = len(yarray)

    # Slice yarray (and xarray) to the x-window before running np.diff.
    # A crossing between samples i-1 and i has x_cross in (x[i-1], x[i]).
    # If x[i] < x_low the crossing is definitely before the window; if
    # x[i-1] > x_high it is definitely after.  Convert the window to an
    # index range (with floor/ceil so boundary crossings are never missed),
    # then keep the precise x_cross check inside the loop.
    if not found_xarray:
        i_low  = (max(0, math.floor((x_low  - xstart) / xdelta))
                  if not math.isinf(x_low)  else 0)
        i_high = (min(n - 1, math.ceil((x_high - xstart) / xdelta))
                  if not math.isinf(x_high) else n - 1)
    else:
        i_low  = (int(np.searchsorted(xarray, x_low,  side="left"))
                  if not math.isinf(x_low)  else 0)
        i_high = (int(np.searchsorted(xarray, x_high, side="right")) - 1
                  if not math.isinf(x_high) else n - 1)
        i_low  = max(0,     i_low)
        i_high = min(n - 1, i_high)

    # i_offset is added back to local indices after slicing
    i_offset = i_low
    y_slice  = yarray[i_low: i_high + 1]

    level_crossings = np.diff(y_slice > ylevel, prepend=False)
    locations = np.argwhere(level_crossings)
    if len(locations.shape) != 2:
        raise RuntimeError("locations shape should be 2")
    locations = locations[:, 0]

    indexes = []
    xvalues = []

    for i_local in locations:
        if i_local == 0 and i_offset == 0:
            continue

        i = i_local + i_offset   # global index in yarray

        y0 = yarray[i - 1]
        y1 = yarray[i]

        if f == "level+":
            if y1 <= y0:
                continue
        elif f == "level-":
            if y1 >= y0:
                continue

        if found_xarray:
            xa = xarray[i - 1]
            xb = xarray[i]
            dx = xb - xa
        else:
            xa = xstart + (i - 1) * xdelta
            xb = xstart + i * xdelta
            dx = xdelta

        # Interpolated crossing x — used for window check and nearest-sample logic
        dy = y1 - y0
        m = dy / dx
        b = y1 - m * xb
        x_cross = (ylevel - b) / m

        # Precise window check (guards against floor/ceil boundary cases)
        if not (x_low <= x_cross <= x_high):
            continue

        if abs(x_cross - xa) <= abs(x_cross - xb):
            idx = int(orig_indices[i - 1]) if orig_indices is not None else i - 1
            indexes.append(idx)
            xvalues.append(x_cross if x_interp else xa)
        else:
            idx = int(orig_indices[i]) if orig_indices is not None else i
            indexes.append(idx)
            xvalues.append(x_cross if x_interp else xb)

    if backward:
        indexes.reverse()
        xvalues.reverse()

    return (np.array(indexes), np.array(xvalues))


# =========================================================================
# Event detection
# =========================================================================

_VALID_EVENT_POLARITIES: frozenset[str] = frozenset({"negative", "positive"})
_VALID_SLIDING_MODES: frozenset[str] = frozenset({"threshold", "nstdv"})


def find_events_sliding_baseline(
    yarray: np.ndarray,
    xstart: float,
    xdelta: float,
    polarity: str,
    mode: str,
    threshold: float,
    baseline_avg: float,
    baseline_dt: float,
    refractory: float = 0.0,
    xbgn: float | None = None,
    xend: float | None = None,
    max_events: int = 0,
) -> list[float]:
    """Detect events using a sliding baseline (Kudoh & Taguchi 2002).

    Forward sliding-window search. At each baseline midpoint t0, computes
    the local average (and optionally stdv) within a window of size
    *baseline_avg*, then checks whether the data at t0 + *baseline_dt*
    crosses the detection level. After a detection, the search resumes at
    t_event + *refractory*.

    Args:
        yarray: 1-D numpy array of y-values.
        xstart: X-value of the first sample.
        xdelta: Sample interval (must be > 0).
        polarity: ``"negative"`` (detect downward deflections) or
            ``"positive"`` (detect upward deflections).
        mode: ``"threshold"`` (fixed amplitude) or ``"nstdv"`` (N×stdv of
            local baseline).
        threshold: Threshold magnitude (>= 0). For mode="threshold" the
            detection level is Y_avg ± threshold; for mode="nstdv" it is
            Y_avg ± threshold × Y_stdv.
        baseline_avg: Baseline averaging window size (x-units). Set to 0
            to use the single point at t0 instead of a windowed average.
        baseline_dt: Time from t0 to the candidate detection point (x-units,
            > 0). Equivalent to *wi* of Kudoh & Taguchi 2002.
        refractory: Minimum time between events (x-units, >= 0). After each
            detection the next search begins at t_event + refractory.
            Default 0.0 (next t0 = t_event, no enforced gap).
        xbgn: Search start (x-units). Default None (start of array).
        xend: Search end (x-units). Default None (end of array).
        max_events: Stop after finding this many events. 0 means no limit
            (default).

    Returns:
        List of detected event x-times (floats).

    Raises:
        TypeError: If yarray is not a numpy ndarray or numeric params have
            wrong types (bool rejected).
        ValueError: If polarity or mode are invalid, or threshold/baseline_dt
            violate their bounds.
    """
    if not isinstance(yarray, np.ndarray):
        raise TypeError(nmu.type_error_str(yarray, "yarray", "numpy.ndarray"))
    if not isinstance(polarity, str) or polarity not in _VALID_EVENT_POLARITIES:
        raise ValueError(
            "polarity must be one of %s, got %r"
            % (sorted(_VALID_EVENT_POLARITIES), polarity)
        )
    if not isinstance(mode, str) or mode not in _VALID_SLIDING_MODES:
        raise ValueError(
            "mode must be one of %s, got %r"
            % (sorted(_VALID_SLIDING_MODES), mode)
        )
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(nmu.type_error_str(threshold, "threshold", "float"))
    threshold = float(threshold)
    if threshold < 0:
        raise ValueError("threshold must be >= 0, got %g" % threshold)
    if isinstance(xstart, bool) or not isinstance(xstart, (int, float)):
        raise TypeError(nmu.type_error_str(xstart, "xstart", "float"))
    xstart = float(xstart)
    if isinstance(xdelta, bool) or not isinstance(xdelta, (int, float)):
        raise TypeError(nmu.type_error_str(xdelta, "xdelta", "float"))
    xdelta = float(xdelta)
    if xdelta <= 0:
        raise ValueError("xdelta must be > 0, got %g" % xdelta)
    if isinstance(baseline_avg, bool) or not isinstance(baseline_avg, (int, float)):
        raise TypeError(nmu.type_error_str(baseline_avg, "baseline_avg", "float"))
    baseline_avg = float(baseline_avg)
    if baseline_avg < 0:
        raise ValueError("baseline_avg must be >= 0, got %g" % baseline_avg)
    if mode == "nstdv" and baseline_avg == 0:
        raise ValueError("baseline_avg must be > 0 when mode='nstdv' (required to compute stdv)")
    if isinstance(baseline_dt, bool) or not isinstance(baseline_dt, (int, float)):
        raise TypeError(nmu.type_error_str(baseline_dt, "baseline_dt", "float"))
    baseline_dt = float(baseline_dt)
    if baseline_dt <= 0:
        raise ValueError("baseline_dt must be > 0, got %g" % baseline_dt)
    if isinstance(refractory, bool) or not isinstance(refractory, (int, float)):
        raise TypeError(nmu.type_error_str(refractory, "refractory", "float"))
    refractory = float(refractory)
    if refractory < 0:
        raise ValueError("refractory must be >= 0, got %g" % refractory)

    n = len(yarray)
    if n == 0:
        return []

    i0 = (0 if xbgn is None or (isinstance(xbgn, float) and math.isinf(xbgn))
          else max(0, round((float(xbgn) - xstart) / xdelta)))
    i1 = (n - 1 if xend is None or (isinstance(xend, float) and math.isinf(xend))
          else min(n - 1, round((float(xend) - xstart) / xdelta)))
    if i0 > i1:
        return []

    avg_ihalf = max(0, round(baseline_avg / xdelta / 2)) if baseline_avg > 0 else 0
    dt_pts   = max(1, round(baseline_dt / xdelta))
    ref_pts  = max(0, round(refractory / xdelta))

    yf = yarray.astype(float, copy=False)
    neg      = (polarity == "negative")
    use_nstdv = (mode == "nstdv")

    events: list[float] = []
    t0_idx = i0

    while t0_idx <= i1:
        det_idx = t0_idx + dt_pts
        if det_idx > i1 or det_idx >= n:
            break

        if avg_ihalf == 0:
            y_avg = yf[t0_idx]
            y_std = 0.0
        else:
            ibsl_lo = max(0, t0_idx - avg_ihalf)
            ibsl_hi = min(n - 1, t0_idx + avg_ihalf)
            yf_window = yf[ibsl_lo: ibsl_hi + 1]
            y_avg = float(np.mean(yf_window))
            y_std = float(np.std(yf_window)) if use_nstdv else 0.0

        nstdv = threshold
        det_offset = (nstdv * y_std) if use_nstdv else threshold
        det_level = (y_avg - det_offset) if neg else (y_avg + det_offset)
        y_det = yf[det_idx]
        crossed = (y_det < det_level) if neg else (y_det > det_level)

        if crossed:
            events.append(xstart + det_idx * xdelta)
            if max_events > 0 and len(events) >= max_events:
                break
            t0_idx = det_idx + max(1, ref_pts)
        else:
            t0_idx += 1

    return events


def find_event_onset(
    yarray: np.ndarray,
    xstart: float,
    xdelta: float,
    t_event: float,
    polarity: str,
    avg: float,
    nstdv: float,
    limit: float,
) -> float | None:
    """Backward sliding-window search for event onset (Kudoh & Taguchi 2002).

    Slides a window backward from *t_event*. At each position, computes the
    local mean and standard deviation, then checks whether the rightmost
    window point crosses the dynamic level ``Y_avg - nstdv × Y_stdv``
    (negative events) or ``Y_avg + nstdv × Y_stdv`` (positive events).
    The first qualifying position gives the onset time.

    Args:
        yarray: 1-D numpy array of y-values.
        xstart: X-value of the first sample.
        xdelta: Sample interval (must be > 0).
        t_event: X-time of the detected event (start of backward search).
        polarity: ``"negative"`` or ``"positive"``.
        avg: Sliding window size (x-units, >= 0). 0 uses a 1-sample window.
        nstdv: Number of standard deviations for the detection level (>= 0).
        limit: Maximum backward search distance from t_event (x-units, > 0).

    Returns:
        X-time of onset (float), or None if not found within *limit*.

    Raises:
        TypeError: If yarray is not a numpy ndarray or params have wrong types.
        ValueError: If polarity is invalid or limit <= 0.
    """
    if not isinstance(yarray, np.ndarray):
        raise TypeError(nmu.type_error_str(yarray, "yarray", "numpy.ndarray"))
    if not isinstance(polarity, str) or polarity not in _VALID_EVENT_POLARITIES:
        raise ValueError(
            "polarity must be one of %s, got %r"
            % (sorted(_VALID_EVENT_POLARITIES), polarity)
        )
    if isinstance(xdelta, bool) or not isinstance(xdelta, (int, float)):
        raise TypeError(nmu.type_error_str(xdelta, "xdelta", "float"))
    xdelta = float(xdelta)
    if xdelta <= 0:
        raise ValueError("xdelta must be > 0, got %g" % xdelta)
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        raise TypeError(nmu.type_error_str(limit, "limit", "float"))
    limit = float(limit)
    if limit <= 0:
        raise ValueError("limit must be > 0, got %g" % limit)

    n = len(yarray)
    if n == 0:
        return None

    xstart = float(xstart)
    event_idx   = min(n - 1, max(0, round((float(t_event) - xstart) / xdelta)))
    avg_pts     = max(1, round(float(avg) / xdelta)) if avg > 0 else 1
    limit_pts   = max(1, round(limit / xdelta))
    istop        = max(0, event_idx - limit_pts)

    yf  = yarray.astype(float, copy=False)
    neg = (polarity == "negative")

    for r in range(event_idx, istop - 1, -1):
        ilo     = max(0, r - avg_pts + 1)
        yf_window = yf[ilo: r + 1]
        y_avg  = float(np.mean(yf_window))
        y_std  = float(np.std(yf_window))
        y_det  = (y_avg - nstdv * y_std) if neg else (y_avg + nstdv * y_std)
        if neg:
            if yf[r] > y_det:
                return xstart + r * xdelta
        else:
            if yf[r] < y_det:
                return xstart + r * xdelta

    return None


def find_event_peak(
    yarray: np.ndarray,
    xstart: float,
    xdelta: float,
    t_event: float,
    polarity: str,
    avg: float,
    nstdv: float,
    limit: float,
) -> float | None:
    """Forward sliding-window search for event peak (Kudoh & Taguchi 2002).

    Slides a window forward from *t_event*. At each position, computes the
    local mean and standard deviation, then checks whether the leftmost
    window point crosses the dynamic level ``Y_avg - nstdv × Y_stdv``
    (negative events) or ``Y_avg + nstdv × Y_stdv`` (positive events).
    The first qualifying position gives the peak time.

    Args:
        yarray: 1-D numpy array of y-values.
        xstart: X-value of the first sample.
        xdelta: Sample interval (must be > 0).
        t_event: X-time of the detected event (start of forward search).
        polarity: ``"negative"`` or ``"positive"``.
        avg: Sliding window size (x-units, >= 0). 0 uses a 1-sample window.
        nstdv: Number of standard deviations for the detection level (>= 0).
        limit: Maximum forward search distance from t_event (x-units, > 0).

    Returns:
        X-time of peak (float), or None if not found within *limit*.

    Raises:
        TypeError: If yarray is not a numpy ndarray or params have wrong types.
        ValueError: If polarity is invalid or limit <= 0.
    """
    if not isinstance(yarray, np.ndarray):
        raise TypeError(nmu.type_error_str(yarray, "yarray", "numpy.ndarray"))
    if not isinstance(polarity, str) or polarity not in _VALID_EVENT_POLARITIES:
        raise ValueError(
            "polarity must be one of %s, got %r"
            % (sorted(_VALID_EVENT_POLARITIES), polarity)
        )
    if isinstance(xdelta, bool) or not isinstance(xdelta, (int, float)):
        raise TypeError(nmu.type_error_str(xdelta, "xdelta", "float"))
    xdelta = float(xdelta)
    if xdelta <= 0:
        raise ValueError("xdelta must be > 0, got %g" % xdelta)
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        raise TypeError(nmu.type_error_str(limit, "limit", "float"))
    limit = float(limit)
    if limit <= 0:
        raise ValueError("limit must be > 0, got %g" % limit)

    n = len(yarray)
    if n == 0:
        return None

    xstart = float(xstart)
    event_idx = min(n - 1, max(0, round((float(t_event) - xstart) / xdelta)))
    avg_pts   = max(1, round(float(avg) / xdelta)) if avg > 0 else 1
    limit_pts = max(1, round(limit / xdelta))
    istop      = min(n - 1, event_idx + limit_pts)

    yf  = yarray.astype(float, copy=False)
    neg = (polarity == "negative")

    for l in range(event_idx, istop + 1):
        ihi     = min(n - 1, l + avg_pts - 1)
        yf_window = yf[l: ihi + 1]
        y_avg  = float(np.mean(yf_window))
        y_std  = float(np.std(yf_window))
        y_det  = (y_avg - nstdv * y_std) if neg else (y_avg + nstdv * y_std)
        if neg:
            if yf[l] < y_det:
                return xstart + l * xdelta
        else:
            if yf[l] > y_det:
                return xstart + l * xdelta

    return None


# =========================================================================
# Linear regression
# =========================================================================


def linear_regression(
    yarray: np.ndarray,
    xarray: np.ndarray | None = None,
    xstart: float = 0,
    xdelta: float = 1,
    ignore_nans: bool = True,
) -> tuple:
    """Fit a linear regression line to y-data using ``numpy.polyfit``.

    Args:
        yarray:      1-D numpy array of yvalues.
        xarray:      Optional 1-D numpy array of xvalues (same size as
                     *yarray*). Used instead of *xstart*/*xdelta* when
                     provided.
        xstart:      X-scale start value; used to build a uniform x-array
                     when *xarray* is None.
        xdelta:      X-scale sample interval; used to build a uniform
                     x-array when *xarray* is None.
        ignore_nans: If True (default), NaN samples are removed from both
                     arrays before fitting.

    Returns:
        Tuple ``(m, b)`` where *m* is the slope and *b* is the y-intercept.

    Raises:
        TypeError:  If *yarray* or *xarray* is not a numpy ndarray, or
                    *xstart*/*xdelta* is not a number.
        ValueError: If *xstart*/*xdelta* is inf/nan, or array sizes differ.
    """
    if not isinstance(yarray, np.ndarray):
        raise TypeError(nmu.type_error_str(yarray, "yarray", "NumPy.ndarray"))

    found_xarray = False

    if xarray is None:
        pass
    elif isinstance(xarray, np.ndarray):
        if xarray.size == yarray.size:
            found_xarray = True
        else:
            raise ValueError(
                "x-y paired NumPy arrays have different size: %s != %s"
                % (xarray.size, yarray.size)
            )
    else:
        raise TypeError(nmu.type_error_str(xarray, "xarray", "NumPy.ndarray"))

    if not found_xarray:
        if isinstance(xstart, float):
            if math.isinf(xstart) or math.isnan(xstart):
                raise ValueError("xstart: '%s'" % xstart)
        elif isinstance(xstart, int) and not isinstance(xstart, bool):
            pass
        else:
            raise TypeError(nmu.type_error_str(xstart, "xstart", "float"))

        if isinstance(xdelta, float):
            if math.isinf(xdelta) or math.isnan(xdelta):
                raise ValueError("xdelta: '%s'" % xdelta)
        elif isinstance(xdelta, int) and not isinstance(xdelta, bool):
            pass
        else:
            raise TypeError(nmu.type_error_str(xdelta, "xdelta", "float"))

        xbgn = xstart
        xend = xstart + (yarray.size - 1) * xdelta
        xarray = np.linspace(xbgn, xend, yarray.size)

    if ignore_nans:
        mask = ~np.isnan(yarray)
        xarray = xarray[mask]
        yarray = yarray[mask]

    m, b = np.polyfit(xarray, yarray, deg=1)

    return (m, b)


# =========================================================================
# Smooth functions
# =========================================================================

_VALID_SMOOTH_METHODS: frozenset[str] = frozenset({"boxcar", "binomial", "savgol"})


def smooth_boxcar(
    y: np.ndarray,
    window: int,
    passes: int = 1,
) -> np.ndarray:
    """Boxcar (moving average) smooth via ``np.convolve``.

    Args:
        y: 1-D numpy array of yvalues.
        window: Kernel width in points. Must be an odd integer >= 3.
        passes: Number of times to apply the kernel. Must be >= 1. Default 1.

    Returns:
        Smoothed copy of *y* with the same length (``mode='same'``).

    Raises:
        TypeError: If *y* is not a numpy ndarray, or *window*/*passes* are
            not integers (bool rejected).
        ValueError: If *window* is not odd, < 3, or *passes* < 1.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(nmu.type_error_str(y, "y", "numpy.ndarray"))
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError(nmu.type_error_str(window, "window", "int"))
    if window < 3:
        raise ValueError("window must be >= 3, got %d" % window)
    if window % 2 == 0:
        raise ValueError("window must be odd, got %d" % window)
    if isinstance(passes, bool) or not isinstance(passes, int):
        raise TypeError(nmu.type_error_str(passes, "passes", "int"))
    if passes < 1:
        raise ValueError("passes must be >= 1, got %d" % passes)
    kernel = np.ones(window) / window
    result = y.copy().astype(float)
    for _ in range(passes):
        result = np.convolve(result, kernel, mode='same')
    return result


def smooth_binomial(
    y: np.ndarray,
    passes: int = 1,
) -> np.ndarray:
    """Binomial smooth: apply 3-point binomial kernel via ``np.convolve``.

    Args:
        y: 1-D numpy array of yvalues.
        passes: Number of times to apply the 3-point binomial kernel.
            Must be >= 1. More passes produce a wider effective smooth.

    Returns:
        Smoothed copy of *y* with the same length (``mode='same'``).

    Raises:
        TypeError: If *y* is not a numpy ndarray, or *passes* is not an
            integer (bool rejected).
        ValueError: If *passes* < 1.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(nmu.type_error_str(y, "y", "numpy.ndarray"))
    if isinstance(passes, bool) or not isinstance(passes, int):
        raise TypeError(nmu.type_error_str(passes, "passes", "int"))
    if passes < 1:
        raise ValueError("passes must be >= 1, got %d" % passes)
    kernel = np.array([0.25, 0.5, 0.25]) # 3-point binomial filter
    result = y.copy().astype(float)
    for _ in range(passes):
        result = np.convolve(result, kernel, mode='same')
    return result


def smooth_savgol(
    y: np.ndarray,
    window: int,
    polyorder: int = 2,
) -> np.ndarray:
    """Savitzky-Golay smooth via ``scipy.signal.savgol_filter``.

    Args:
        y: 1-D numpy array of yvalues.
        window: Kernel width in points. Must be an odd integer and
            >= *polyorder* + 2.
        polyorder: Polynomial order. Must be an integer >= 1 and
            < *window*. Default 2.

    Returns:
        Smoothed copy of *y* with the same length.

    Raises:
        TypeError: If *y* is not a numpy ndarray, or any numeric
            parameter is not an integer (bool rejected).
        ValueError: If *window* is not odd, too small, or *polyorder* is
            out of range.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(nmu.type_error_str(y, "y", "numpy.ndarray"))
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError(nmu.type_error_str(window, "window", "int"))
    if window < 3:
        raise ValueError("window must be >= 3, got %d" % window)
    if window % 2 == 0:
        raise ValueError("window must be odd, got %d" % window)
    if isinstance(polyorder, bool) or not isinstance(polyorder, int):
        raise TypeError(nmu.type_error_str(polyorder, "polyorder", "int"))
    if polyorder < 1:
        raise ValueError("polyorder must be >= 1, got %d" % polyorder)
    if polyorder >= window:
        raise ValueError(
            "polyorder (%d) must be < window (%d)" % (polyorder, window)
        )
    from scipy.signal import savgol_filter
    return savgol_filter(y.copy().astype(float), window, polyorder)


# =========================================================================
# Filter functions
# =========================================================================

_VALID_FILTER_TYPES: frozenset[str] = frozenset({"butterworth", "bessel", "notch"})
_VALID_FILTER_BTYPES: frozenset[str] = frozenset({"low", "high", "bandpass"})


def filter_butterworth(
    y: np.ndarray,
    cutoff: float | list[float],
    sample_rate: float,
    order: int = 4,
    btype: str = "low",
) -> np.ndarray:
    """Butterworth filter via ``scipy.signal.butter`` + ``sosfiltfilt``.

    Applies a zero-phase Butterworth filter (forward-backward via
    ``sosfiltfilt``), which introduces no phase distortion.  Suitable for
    general-purpose low-pass, high-pass, and band-pass filtering.

    Args:
        y: 1-D numpy array of y-values.
        cutoff: Cutoff frequency in Hz.  For ``btype='bandpass'`` supply a
            two-element list ``[low_hz, high_hz]``.
        sample_rate: Sample rate in Hz (must be > 0).
        order: Filter order (int >= 1). Default 4.
        btype: ``'low'``, ``'high'``, or ``'bandpass'``. Default ``'low'``.

    Returns:
        Filtered copy of *y* with the same length.

    Raises:
        TypeError: If *y* is not a numpy ndarray or numeric params have
            wrong types (bool rejected for int params).
        ValueError: If *sample_rate* <= 0, *order* < 1, *btype* is invalid,
            or *cutoff* is at or above the Nyquist frequency.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(nmu.type_error_str(y, "y", "numpy.ndarray"))
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
        raise TypeError(nmu.type_error_str(sample_rate, "sample_rate", "float"))
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0, got %g" % sample_rate)
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError(nmu.type_error_str(order, "order", "int"))
    if order < 1:
        raise ValueError("order must be >= 1, got %d" % order)
    if not isinstance(btype, str) or btype not in _VALID_FILTER_BTYPES:
        raise ValueError(
            "btype must be one of %s, got %r" % (sorted(_VALID_FILTER_BTYPES), btype)
        )
    if isinstance(cutoff, bool):
        raise TypeError(nmu.type_error_str(cutoff, "cutoff", "float or list"))
    if isinstance(cutoff, (int, float)):
        if cutoff <= 0:
            raise ValueError("cutoff must be > 0, got %g" % cutoff)
    elif not isinstance(cutoff, list):
        raise TypeError(nmu.type_error_str(cutoff, "cutoff", "float or list"))
    from scipy.signal import butter, sosfiltfilt
    nyq = sample_rate / 2.0
    if isinstance(cutoff, (list, np.ndarray)):
        norm = [c / nyq for c in cutoff]
    else:
        norm = cutoff / nyq
    sos = butter(order, norm, btype=btype, output="sos")
    return sosfiltfilt(sos, y.astype(float))


def filter_bessel(
    y: np.ndarray,
    cutoff: float | list[float],
    sample_rate: float,
    order: int = 4,
    btype: str = "low",
) -> np.ndarray:
    """Bessel filter via ``scipy.signal.bessel`` + ``sosfiltfilt``.

    Applies a zero-phase Bessel filter (forward-backward via
    ``sosfiltfilt``).  Bessel filters have maximally flat group delay,
    which preserves waveform shape — preferred when timing is critical
    (e.g. spike kinetics, EPSC rise times).

    Args:
        y: 1-D numpy array of y-values.
        cutoff: Cutoff frequency in Hz.  For ``btype='bandpass'`` supply a
            two-element list ``[low_hz, high_hz]``.
        sample_rate: Sample rate in Hz (must be > 0).
        order: Filter order (int >= 1). Default 4.
        btype: ``'low'``, ``'high'``, or ``'bandpass'``. Default ``'low'``.

    Returns:
        Filtered copy of *y* with the same length.

    Raises:
        TypeError: If *y* is not a numpy ndarray or numeric params have
            wrong types (bool rejected for int params).
        ValueError: If *sample_rate* <= 0, *order* < 1, *btype* is invalid,
            or *cutoff* is at or above the Nyquist frequency.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(nmu.type_error_str(y, "y", "numpy.ndarray"))
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
        raise TypeError(nmu.type_error_str(sample_rate, "sample_rate", "float"))
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0, got %g" % sample_rate)
    if isinstance(order, bool) or not isinstance(order, int):
        raise TypeError(nmu.type_error_str(order, "order", "int"))
    if order < 1:
        raise ValueError("order must be >= 1, got %d" % order)
    if not isinstance(btype, str) or btype not in _VALID_FILTER_BTYPES:
        raise ValueError(
            "btype must be one of %s, got %r" % (sorted(_VALID_FILTER_BTYPES), btype)
        )
    if isinstance(cutoff, bool):
        raise TypeError(nmu.type_error_str(cutoff, "cutoff", "float or list"))
    if isinstance(cutoff, (int, float)):
        if cutoff <= 0:
            raise ValueError("cutoff must be > 0, got %g" % cutoff)
    elif not isinstance(cutoff, list):
        raise TypeError(nmu.type_error_str(cutoff, "cutoff", "float or list"))
    from scipy.signal import bessel, sosfiltfilt
    nyq = sample_rate / 2.0
    if isinstance(cutoff, (list, np.ndarray)):
        norm = [c / nyq for c in cutoff]
    else:
        norm = cutoff / nyq
    sos = bessel(order, norm, btype=btype, output="sos", norm="phase")
    return sosfiltfilt(sos, y.astype(float))


def filter_notch(
    y: np.ndarray,
    freq: float,
    sample_rate: float,
    q: float = 30.0,
) -> np.ndarray:
    """Notch (band-stop) filter via ``scipy.signal.iirnotch`` + ``filtfilt``.

    Removes a narrow frequency band centred on *freq* (e.g. 50 or 60 Hz
    mains interference).  Applies zero-phase filtering via ``filtfilt``.

    Args:
        y: 1-D numpy array of y-values.
        freq: Centre frequency to remove, in Hz (must be > 0 and < Nyquist).
        sample_rate: Sample rate in Hz (must be > 0).
        q: Quality factor — ratio of centre frequency to bandwidth.
            Higher values give a narrower notch. Default 30.

    Returns:
        Filtered copy of *y* with the same length.

    Raises:
        TypeError: If *y* is not a numpy ndarray or numeric params have
            wrong types (bool rejected).
        ValueError: If *sample_rate* <= 0, *freq* <= 0 or >= Nyquist,
            or *q* <= 0.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(nmu.type_error_str(y, "y", "numpy.ndarray"))
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
        raise TypeError(nmu.type_error_str(sample_rate, "sample_rate", "float"))
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0, got %g" % sample_rate)
    if isinstance(freq, bool) or not isinstance(freq, (int, float)):
        raise TypeError(nmu.type_error_str(freq, "freq", "float"))
    nyq = sample_rate / 2.0
    if freq <= 0 or freq >= nyq:
        raise ValueError(
            "freq must be > 0 and < Nyquist (%.6g Hz), got %g" % (nyq, freq)
        )
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        raise TypeError(nmu.type_error_str(q, "q", "float"))
    if q <= 0:
        raise ValueError("q must be > 0, got %g" % q)
    from scipy.signal import iirnotch, sosfiltfilt, tf2sos
    b, a = iirnotch(freq / nyq, q)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, y.astype(float))


# =========================================================================
# Resample and interpolate
# =========================================================================


def resample(
    y: np.ndarray,
    old_delta: float,
    new_delta: float,
) -> np.ndarray:
    """Resample a 1-D array to a new sample interval using polyphase filtering.

    Wraps ``scipy.signal.resample_poly`` which applies an anti-aliasing FIR
    filter before decimation, making it appropriate for both upsampling and
    downsampling.  The ratio ``old_delta / new_delta`` is approximated as a
    rational number (numerator / denominator, limited to 1000) to satisfy
    ``resample_poly``'s integer ``up`` / ``down`` requirement.

    Args:
        y: 1-D numpy array of yvalues.
        old_delta: Current sample interval (any consistent units).
        new_delta: Desired sample interval (same units as *old_delta*).

    Returns:
        Resampled copy of *y*.  Length will be approximately
        ``len(y) * old_delta / new_delta``.

    Raises:
        TypeError: If *y* is not a numpy ndarray, or *old_delta*/*new_delta*
            are not numeric.
        ValueError: If *old_delta* or *new_delta* are <= 0.
    """
    from fractions import Fraction  # noqa: PLC0415

    from scipy.signal import resample_poly  # noqa: PLC0415

    if not isinstance(y, np.ndarray):
        raise TypeError(nmu.type_error_str(y, "y", "numpy.ndarray"))
    if not isinstance(old_delta, (int, float)) or isinstance(old_delta, bool):
        raise TypeError(nmu.type_error_str(old_delta, "old_delta", "float"))
    if not isinstance(new_delta, (int, float)) or isinstance(new_delta, bool):
        raise TypeError(nmu.type_error_str(new_delta, "new_delta", "float"))
    if old_delta <= 0:
        raise ValueError("old_delta must be > 0, got %g" % old_delta)
    if new_delta <= 0:
        raise ValueError("new_delta must be > 0, got %g" % new_delta)
    frac = Fraction(old_delta / new_delta).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    return resample_poly(y.copy().astype(float), up, down)


_VALID_INTERPOLATE_METHODS: frozenset[str] = frozenset({"linear", "cubic"})


def interpolate(
    y: np.ndarray,
    x_old: np.ndarray,
    x_new: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate a 1-D array from one x-axis to another.

    Uses ``numpy.interp`` for linear interpolation and
    ``scipy.interpolate.CubicSpline`` for cubic interpolation.  Values
    outside the range of *x_old* are filled with ``NaN``.

    Args:
        y: 1-D numpy array of yvalues corresponding to *x_old*.
        x_old: 1-D numpy array of original x-positions (must be strictly
            increasing).
        x_new: 1-D numpy array of target x-positions.
        method: ``"linear"`` (default) or ``"cubic"``.

    Returns:
        1-D numpy array of interpolated yvalues at *x_new* positions.

    Raises:
        TypeError: If any array argument is not a numpy ndarray.
        ValueError: If *method* is not ``"linear"`` or ``"cubic"``.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(nmu.type_error_str(y, "y", "numpy.ndarray"))
    if not isinstance(x_old, np.ndarray):
        raise TypeError(nmu.type_error_str(x_old, "x_old", "numpy.ndarray"))
    if not isinstance(x_new, np.ndarray):
        raise TypeError(nmu.type_error_str(x_new, "x_new", "numpy.ndarray"))
    if method not in _VALID_INTERPOLATE_METHODS:
        raise ValueError(
            "method must be one of %s, got %r"
            % (sorted(_VALID_INTERPOLATE_METHODS), method)
        )
    if method == "linear":
        return np.interp(x_new, x_old, y, left=np.nan, right=np.nan)
    # cubic
    from scipy.interpolate import CubicSpline  # noqa: PLC0415

    cs = CubicSpline(x_old, y, extrapolate=False)
    return cs(x_new).astype(float)


# =========================================================================
# Stats functions
# =========================================================================


def histogram(
    y: np.ndarray,
    bins: int | list = 10,
    xrange: tuple | None = None,
    density: bool = False,
) -> dict:
    """Thin wrapper around ``np.histogram`` that excludes NaN and Inf values.

    Args:
        y: 1-D numpy array of values.
        bins: Number of equal-width bins (int) or explicit bin edges (list).
            Default 10.
        xrange: ``(min, max)`` tuple to restrict the data range. Default None
            (full data range).
        density: If True, return probability density instead of counts.
            Default False.

    Returns:
        Dict with keys:

        - ``"counts"``: bin counts or density values (numpy array).
        - ``"edges"``: bin edge values, length = bins + 1 (numpy array).

    Raises:
        TypeError: If *y* is not a numpy ndarray.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError(nmu.type_error_str(y, "y", "numpy.ndarray"))
    arr = y.astype(float)
    arr = arr[np.isfinite(arr)]
    counts, edges = np.histogram(arr, bins=bins, range=xrange, density=density)
    return {"counts": counts, "edges": edges}


def ks_test(
    y1: np.ndarray,
    y2: np.ndarray,
    alpha: float = 0.05,
    method: str = "auto",
) -> dict:
    """Two-sample Kolmogorov-Smirnov test on two 1-D arrays.

    NaN and Inf values are excluded before testing, matching the Igor
    ``NMKSTest()`` behaviour of removing NaNs before sorting.

    This mirrors the Igor NeuroMatic ``NMKSTest()`` function
    (``NM_StatsTabKolmogorov.ipf``), originally written by Dr. Angus
    Silver (UCL) based on *Numerical Recipes*, but now uses 
    ``scipy.stats.ks_2samp`` for p-value calculation.

    Args:
        y1: First 1-D numpy array of values.
        y2: Second 1-D numpy array of values.
        alpha: Significance level; ``significant`` is True when
            ``pvalue <= alpha``. Default 0.05.
        method: P-value calculation method forwarded to
            ``scipy.stats.ks_2samp``. One of ``"auto"`` (default),
            ``"exact"``, or ``"asymp"``.

    Returns:
        Dict with keys:

        - ``"d"``: KS statistic — max |CDF1 - CDF2|.
        - ``"pvalue"``: p-value.
        - ``"alpha"``: significance level (echoed from param).
        - ``"significant"``: True if ``pvalue <= alpha``.
        - ``"message"``: ``"different populations"`` or ``"same population"``.
        - ``"n1"``: sample size of y1 after NaN/Inf removal.
        - ``"n2"``: sample size of y2 after NaN/Inf removal.
        - ``"sort1"``: sorted finite values of y1 (for empirical CDF).
        - ``"ecdf1"``: empirical CDF values for y1.
        - ``"sort2"``: sorted finite values of y2 (for empirical CDF).
        - ``"ecdf2"``: empirical CDF values for y2.

    Raises:
        TypeError: If *y1* or *y2* are not numpy ndarrays.
        ImportError: If scipy is not installed.
    """
    from scipy.stats import ks_2samp  # noqa: PLC0415

    if not isinstance(y1, np.ndarray):
        raise TypeError(nmu.type_error_str(y1, "y1", "numpy.ndarray"))
    if not isinstance(y2, np.ndarray):
        raise TypeError(nmu.type_error_str(y2, "y2", "numpy.ndarray"))
    arr1 = y1.astype(float)
    arr2 = y2.astype(float)
    arr1 = arr1[np.isfinite(arr1)]
    arr2 = arr2[np.isfinite(arr2)]
    stat, pvalue = ks_2samp(arr1, arr2, method=method)
    significant = bool(pvalue <= alpha)
    sort1 = np.sort(arr1)
    ecdf1 = np.arange(1, len(sort1) + 1) / len(sort1)
    sort2 = np.sort(arr2)
    ecdf2 = np.arange(1, len(sort2) + 1) / len(sort2)
    return {
        "d":           float(stat),
        "pvalue":      float(pvalue),
        "alpha":       float(alpha),
        "significant": significant,
        "message":     "different populations" if significant else "same population",
        "n1":          int(len(arr1)),
        "n2":          int(len(arr2)),
        "sort1":       sort1,
        "ecdf1":       ecdf1,
        "sort2":       sort2,
        "ecdf2":       ecdf2,
    }


def stability_test(
    y: np.ndarray,
    alpha: float = 0.05,
    min_window: int = 10,
) -> dict:
    """Find the largest stable (trend-free) window in a 1-D array.

    Tests whether consecutive subsets of values have no significant monotonic
    trend using the Spearman rank-order correlation between values
    and their array indices. Searches from the largest possible window down to
    ``min_window``, stopping as soon as the largest stable window is found.

    This mirrors the Igor NeuroMatic ``NMStabilityRankOrderTest()`` function
    (``NM_StatsTabStability.ipf``, pass 1 only), originally written by
    Dr. Angus Silver and Simon Mitchell (UCL), based on *Numerical Recipes in C*,
    but now uses ``scipy.stats.spearmanr`` for p-value calculation.

    NaN and Inf values are excluded before the search; returned indices
    (``start``, ``end``) refer to positions in the original array *y*.

    Args:
        y: 1-D numpy array of values.
        alpha: Significance level. A window is "stable" when its Spearman
            p-value exceeds this threshold. Default 0.05.
        min_window: Minimum window size in data points. Must be >= 3 (scipy
            requires at least 3 points to compute a p-value for non-constant
            data). Default 10.

    Returns:
        Dict with keys:

        - ``"stable"``: True if a stable region was found.
        - ``"start"``: Index into original *y* of first stable point, or None.
        - ``"end"``: Index into original *y* of last stable point
          (inclusive), or None.
        - ``"n"``: Window size (``end - start + 1``), or None.
        - ``"rs"``: Spearman rs at best window, or None.
        - ``"pvalue"``: p-value at best window, or None.
        - ``"alpha"``: significance level (echoed from param).
        - ``"mask"``: Boolean numpy array (length = len(y)), True where the
          stable region falls.

    Raises:
        TypeError: If *y* is not a numpy ndarray.
        ValueError: If *min_window* < 3 or > number of finite data points.
        ImportError: If scipy is not installed.
    """
    import warnings  # noqa: PLC0415

    from scipy.stats import spearmanr  # noqa: PLC0415

    if not isinstance(y, np.ndarray):
        raise TypeError(nmu.type_error_str(y, "y", "numpy.ndarray"))

    arr = y.astype(float)
    finite_mask = np.isfinite(arr)
    finite_idx = np.where(finite_mask)[0]
    arr_clean = arr[finite_mask]
    n = len(arr_clean)

    if min_window < 3:
        raise ValueError("min_window must be >= 3, got %d" % min_window)
    if min_window > n:
        raise ValueError(
            "min_window (%d) > available data points (%d)" % (min_window, n)
        )

    best_start: int | None = None
    best_size: int = 0
    best_rs: float | None = None
    best_pvalue: float = 0.0

    for w in range(n, min_window - 1, -1):
        for i in range(n - w + 1):
            x = np.arange(i, i + w, dtype=float)
            ywin = arr_clean[i: i + w]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rs, pvalue = spearmanr(x, ywin)
            if math.isnan(rs):
                rs, pvalue = 0.0, 1.0
            elif math.isnan(pvalue):
                continue
            if pvalue > alpha and pvalue > best_pvalue:
                best_start = i
                best_size = w
                best_rs = float(rs)
                best_pvalue = float(pvalue)
        if best_start is not None:
            break

    mask = np.zeros(len(arr), dtype=bool)

    if best_start is None:
        return {
            "stable":  False,
            "start":   None,
            "end":     None,
            "n":       None,
            "rs":      None,
            "pvalue":  None,
            "alpha":   float(alpha),
            "mask":    mask,
        }

    best_end = best_start + best_size - 1
    orig_start = int(finite_idx[best_start])
    orig_end = int(finite_idx[best_end])
    mask[finite_idx[best_start: best_end + 1]] = True

    return {
        "stable":  True,
        "start":   orig_start,
        "end":     orig_end,
        "n":       best_size,
        "rs":      best_rs,
        "pvalue":  best_pvalue,
        "alpha":   float(alpha),
        "mask":    mask,
    }


# =========================================================================
# Template matching
# =========================================================================


def match_template(
    data: np.ndarray,
    template: np.ndarray,
    circular: bool = False,
) -> np.ndarray:
    """Sliding template matching (Clements & Bekkers 1997).

    Fits a scaled, offset copy of *template* to each position in *data* using
    ordinary least-squares and returns a detection-criterion array
    (scale / SE) of the same length as *data*.  An event is detected when the
    criterion crosses a threshold (Clements & Bekkers suggest ±4 for synaptic
    events).

    Ports the core algorithm from the Igor XOP ``MatchTemplate.cpp``
    (``DoWave`` function).

    Reference:
        Clements JD, Bekkers JM. Detection of spontaneous synaptic events with
        an optimally scaled template. Biophys J. 1997 Jul;73(1):220-9.
        doi: 10.1016/S0006-3495(97)78062-7.
        PMID: 9199786; PMCID: PMC1180923.

    Note:
        The criterion (scale / SE) is inversely proportional to the template
        amplitude: multiplying the template by ``k`` divides the criterion by
        ``k`` while leaving SE unchanged.  The conventional threshold of ±4
        therefore assumes a template normalized to unit amplitude.  Use a
        template built from an average event waveform and normalize its peak
        to 1, or adjust the threshold to match the template amplitude.

    Args:
        data: 1-D numpy array of recorded values.
        template: 1-D numpy array representing a single-event waveform.
            Must have at least 2 points and be no longer than *data*.
        circular: If True, windows wrap circularly at the end of *data*
            (equivalent to the ``/C`` flag in the Igor XOP).  The number of
            active positions equals ``len(data)``.  If False (default), only
            positions ``0`` to ``len(data) - len(template)`` are computed;
            the remaining tail is set to zero.

    Returns:
        1-D float64 numpy array of length ``len(data)``.  Each value is the
        detection criterion (scale / SE) at that position, or zero where
        the standard error is zero or beyond the valid window range.

    Raises:
        TypeError: If *data* or *template* is not a numpy ndarray.
        ValueError: If *data* or *template* is not 1-D, if *template* has
            fewer than 2 points, or if *template* is longer than *data*.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(nmu.type_error_str(data, "data", "NumPy ndarray"))
    if not isinstance(template, np.ndarray):
        raise TypeError(nmu.type_error_str(template, "template", "NumPy ndarray"))
    if data.ndim != 1:
        raise ValueError("data must be 1-D, got shape %s" % str(data.shape))
    if template.ndim != 1:
        raise ValueError("template must be 1-D, got shape %s" % str(template.shape))

    n = len(data)
    m = len(template)

    if m < 2:
        raise ValueError("template must have at least 2 points, got %d" % m)
    if m > n:
        raise ValueError(
            "template length (%d) exceeds data length (%d)" % (m, n)
        )

    data = data.astype(float, copy=False)
    template = template.astype(float, copy=False)

    tsum = float(np.sum(template))
    tsumsqr = float(np.sum(template ** 2))
    pnts = float(m)
    denom = tsumsqr - tsum * tsum / pnts

    if denom == 0.0:
        return np.zeros(n)

    if circular:
        passes = n
        ext = np.concatenate([data, data[:m - 1]])
    else:
        passes = n - m + 1
        ext = data

    # Sliding window sums via prefix sums
    cs = np.concatenate([[0.0], np.cumsum(ext)])
    dsum = cs[m:m + passes] - cs[:passes]

    cssq = np.concatenate([[0.0], np.cumsum(ext ** 2)])
    dsumsqr = cssq[m:m + passes] - cssq[:passes]

    # Cross-product sum: dtsum[i] = sum(ext[i:i+m] * template)
    dtsum = np.correlate(ext, template, mode="valid")[:passes]

    # Optimal scale and offset (Clements & Bekkers 1997, Eq. 3-4)
    scale = (dtsum - tsum * dsum / pnts) / denom
    offset = (dsum - scale * tsum) / pnts

    # Sum of squared errors (Eq. 5)
    sse = (
        dsumsqr
        + scale ** 2 * tsumsqr
        + pnts * offset ** 2
        - 2.0 * (scale * dtsum + offset * dsum - scale * offset * tsum)
    )

    se = np.sqrt(np.maximum(sse, 0.0) / (pnts - 1.0))
    safe_se = np.where(se == 0.0, 1.0, se)
    detect = np.where(se == 0.0, 0.0, scale / safe_se)

    result = np.zeros(n)
    result[:passes] = detect
    return result


# ---------------------------------------------------------------------------
# Curve-fitting functions
# ---------------------------------------------------------------------------

def _extract_xy_window(
    yarray: np.ndarray,
    xstart: float,
    xdelta: float,
    xarray: np.ndarray | None,
    xbgn: float,
    xend: float,
    ignore_nans: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract x and y within [xbgn, xend], optionally removing NaNs."""
    n = len(yarray)
    if xarray is not None:
        xdata = xarray.astype(float, copy=False)
    else:
        xdata = xstart + np.arange(n) * xdelta

    mask = (xdata >= xbgn) & (xdata <= xend)
    x = xdata[mask]
    y = yarray[mask].astype(float, copy=False)

    if ignore_nans:
        valid = ~np.isnan(y)
        x = x[valid]
        y = y[valid]

    return x, y


def _r2(y: np.ndarray, y_fit: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def fit_line(
    yarray: np.ndarray,
    xstart: float = 0.0,
    xdelta: float = 1.0,
    xarray: np.ndarray | None = None,
    xbgn: float = -math.inf,
    xend: float = math.inf,
    sigma: np.ndarray | None = None,
    ignore_nans: bool = True,
) -> dict:
    """Fit a straight line to *yarray*.

    Without *sigma* uses ``scipy.stats.linregress``.  With *sigma* uses
    ``numpy.polyfit(deg=1, w=1/sigma)`` for weighted least squares.

    Args:
        yarray:  1-D numpy array of y-values.
        xstart:  X-axis start value (uniform spacing). Ignored when *xarray*
                 is provided.
        xdelta:  X-axis sample interval (uniform spacing). Ignored when
                 *xarray* is provided.
        xarray:  Non-uniform x-values array. When provided, *xstart* and
                 *xdelta* are ignored.
        xbgn:    Fit window start (x-units). Default ``-inf``.
        xend:    Fit window end (x-units). Default ``+inf``.
        sigma:   Per-point standard deviations for weighted fitting. When
                 provided weights are ``1/sigma`` and chi-squared uses sigma
                 in the denominator.
        ignore_nans: If True (default), exclude NaN values before fitting.

    Returns:
        Dict with keys ``"slope"``, ``"intercept"``, ``"r2"``,
        ``"chi_sqr"``, ``"n"``.

    Raises:
        ValueError: If fewer than 2 data points remain after windowing/NaN
            removal, or if *sigma* has a different length than the windowed
            y-data.
    """
    x, y = _extract_xy_window(yarray, xstart, xdelta, xarray, xbgn, xend, ignore_nans)
    n = len(x)
    if n < 2:
        raise ValueError("fit_line: need at least 2 data points, got %d" % n)

    if sigma is not None:
        w = 1.0 / np.asarray(sigma, dtype=float)
        if len(w) != n:
            raise ValueError(
                "fit_line: sigma length (%d) != windowed data length (%d)" % (len(w), n)
            )
        try:
            coeffs, cov = np.polyfit(x, y, 1, w=w, cov=True)
            slope_err     = float(np.sqrt(max(0.0, cov[0, 0])))
            intercept_err = float(np.sqrt(max(0.0, cov[1, 1])))
        except np.linalg.LinAlgError:
            coeffs = np.polyfit(x, y, 1, w=w)
            slope_err = intercept_err = float("nan")
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
        y_fit = slope * x + intercept
        r2 = _r2(y, y_fit)
        residuals = y - y_fit
        chi_sqr = float(np.sum((residuals / np.asarray(sigma, dtype=float)) ** 2))
    else:
        from scipy.stats import linregress  # noqa: PLC0415
        res = linregress(x, y)
        slope = float(res.slope)
        intercept = float(res.intercept)
        slope_err     = float(res.stderr)
        intercept_err = float(getattr(res, "intercept_stderr", float("nan")))
        y_fit = slope * x + intercept
        r2 = float(res.rvalue ** 2)
        residuals = y - y_fit
        chi_sqr = float(np.sum(residuals ** 2)) / n

    return {
        "slope":         slope,
        "intercept":     intercept,
        "slope_err":     slope_err,
        "intercept_err": intercept_err,
        "r2":            r2,
        "chi_sqr":       chi_sqr,
        "yfit":          y_fit,
        "residuals":     residuals,
        "x":             x,
        "n":             n,
    }


def fit_poly(
    yarray: np.ndarray,
    xstart: float = 0.0,
    xdelta: float = 1.0,
    xarray: np.ndarray | None = None,
    xbgn: float = -math.inf,
    xend: float = math.inf,
    degree: int = 2,
    sigma: np.ndarray | None = None,
    ignore_nans: bool = True,
) -> dict:
    """Fit a polynomial to *yarray*.

    Uses ``numpy.polyfit``.  Coefficients are stored in ascending order:
    ``coefficients[0]`` is the constant term (C0), ``coefficients[k]`` is
    the coefficient of x^k.

    Args:
        yarray:  1-D numpy array of y-values.
        xstart:  X-axis start value (uniform spacing).
        xdelta:  X-axis sample interval (uniform spacing).
        xarray:  Non-uniform x-values array.
        xbgn:    Fit window start (x-units). Default ``-inf``.
        xend:    Fit window end (x-units). Default ``+inf``.
        degree:  Polynomial degree ≥ 1.
        sigma:   Per-point standard deviations for weighted fitting.
        ignore_nans: If True (default), exclude NaN values before fitting.

    Returns:
        Dict with keys ``"coefficients"`` (list, ascending order),
        ``"degree"``, ``"r2"``, ``"chi_sqr"``, ``"n"``.

    Raises:
        ValueError: If *degree* < 1, fewer than ``degree + 1`` points remain,
            or *sigma* length mismatch.
    """
    if degree < 1:
        raise ValueError("fit_poly: degree must be >= 1, got %d" % degree)
    x, y = _extract_xy_window(yarray, xstart, xdelta, xarray, xbgn, xend, ignore_nans)
    n = len(x)
    if n < degree + 1:
        raise ValueError(
            "fit_poly: need at least %d data points for degree %d, got %d"
            % (degree + 1, degree, n)
        )

    w: np.ndarray | None = None
    if sigma is not None:
        w = 1.0 / np.asarray(sigma, dtype=float)
        if len(w) != n:
            raise ValueError(
                "fit_poly: sigma length (%d) != windowed data length (%d)" % (len(w), n)
            )

    # polyfit returns coefficients highest-order first — reverse to ascending
    try:
        coeffs_desc, cov = np.polyfit(x, y, degree, w=w, cov=True)
        # cov[0,0] = variance of highest-degree coeff; cov[degree,degree] = constant
        coef_errors_asc = [
            float(np.sqrt(max(0.0, cov[degree - k, degree - k])))
            for k in range(degree + 1)
        ]
    except np.linalg.LinAlgError:
        coeffs_desc = np.polyfit(x, y, degree, w=w)
        coef_errors_asc = [float("nan")] * (degree + 1)

    coeffs_asc = list(reversed([float(c) for c in coeffs_desc]))
    y_fit = np.polyval(coeffs_desc, x)
    r2 = _r2(y, y_fit)
    residuals = y - y_fit

    if sigma is not None and w is not None:
        chi_sqr = float(np.sum((residuals / np.asarray(sigma, dtype=float)) ** 2))
    else:
        chi_sqr = float(np.sum(residuals ** 2)) / n

    return {
        "coefficients":  coeffs_asc,
        "coef_errors":   coef_errors_asc,
        "degree":        degree,
        "r2":            r2,
        "chi_sqr":       chi_sqr,
        "yfit":          y_fit,
        "residuals":     residuals,
        "x":             x,
        "n":             n,
    }


def fit_exp(
    yarray: np.ndarray,
    xstart: float = 0.0,
    xdelta: float = 1.0,
    xarray: np.ndarray | None = None,
    xbgn: float = -math.inf,
    xend: float = math.inf,
    x_origin: float = 0.0,
    p0: dict | None = None,
    sigma: np.ndarray | None = None,
    maxfev: int = 10000,
    ignore_nans: bool = True,
) -> dict:
    """Fit a single exponential to *yarray*.

    Model: ``f(x) = A * exp(-(x - X0) / Tau) + Y0``

    where:

    * **A** — amplitude (y-scale units)
    * **Tau** — decay time constant (x-scale units; use ``Tau > 0`` for decay,
      ``Tau < 0`` for growth)
    * **X0** — fixed x-origin (``x_origin``); not fitted
    * **Y0** — y-offset

    *X0* (``x_origin``) shifts the x-origin and is not fitted.  Set it to
    the start of the recording window so that ``Tau`` is measured from that
    point rather than from x = 0.

    Missing initial-parameter keys in *p0* are auto-estimated from the data:
    ``A = y[0] - y[-1]``, ``Tau = (x[-1] - x[0]) / 3``, ``Y0 = y[-1]``.

    Args:
        yarray:    1-D numpy array of y-values.
        xstart:    X-axis start value (uniform spacing).
        xdelta:    X-axis sample interval (uniform spacing).
        xarray:    Non-uniform x-values array; overrides *xstart*/*xdelta*.
        xbgn:      Fit window start (x-units). Default ``-inf``.
        xend:      Fit window end (x-units). Default ``+inf``.
        x_origin:  Fixed x-offset (X0) in the model. Default 0.0.
        p0:        Initial parameter estimates as a dict with optional keys
                   ``"A"``, ``"Tau"``, ``"Y0"``.  Missing keys are
                   auto-estimated.
        sigma:     Per-point standard deviations for weighted fitting.
        maxfev:    Maximum function evaluations for the optimizer. Default 10000.
        ignore_nans: If True (default), exclude NaN values before fitting.

    Returns:
        Dict with keys:

        * ``"A"``, ``"Tau"``, ``"X0"``, ``"Y0"`` — fitted parameters
          (``"X0"`` echoes the fixed ``x_origin`` value)
        * ``"A_err"``, ``"Tau_err"``, ``"Y0_err"`` — one-standard-deviation
          parameter uncertainties from the covariance matrix
        * ``"r2"`` — coefficient of determination
        * ``"chi_sqr"`` — weighted sum of squared residuals (if *sigma*
          supplied) or mean sum of squared residuals
        * ``"yfit"`` — model evaluated at the fit x-points
        * ``"residuals"`` — ``y - yfit`` at the fit x-points
        * ``"x"`` — x-values used for the fit (after windowing / NaN removal)
        * ``"n"`` — number of data points used
        * ``"converged"`` — ``True`` if the optimizer converged

    Raises:
        ValueError: Fewer than 3 data points, or *sigma* length mismatch.
    """
    from scipy.optimize import curve_fit  # noqa: PLC0415

    x, y = _extract_xy_window(yarray, xstart, xdelta, xarray, xbgn, xend, ignore_nans)
    n = len(x)
    if n < 3:
        raise ValueError("fit_exp: need at least 3 data points, got %d" % n)

    if sigma is not None and len(sigma) != n:
        raise ValueError(
            "fit_exp: sigma length (%d) != windowed data length (%d)" % (len(sigma), n)
        )

    p0_dict = p0 or {}
    # Use short-window medians at each end to resist outliers/noise
    _win = max(1, n // 10)
    _y_start = float(np.nanmedian(y[:_win]))
    _y_end   = float(np.nanmedian(y[-_win:]))
    A0  = p0_dict.get("A",  _y_start - _y_end)
    x_span = float(x[-1] - x[0])
    B0  = p0_dict.get("Tau",  x_span / 3.0 if x_span > 0 else 1.0)
    Y0_0 = p0_dict.get("Y0", _y_end)

    def _model(xv: np.ndarray, A: float, B: float, Y0: float) -> np.ndarray:
        return A * np.exp(-(xv - x_origin) / B) + Y0

    converged = True
    try:
        popt, pcov = curve_fit(
            _model, x, y,
            p0=[A0, B0, Y0_0],
            sigma=sigma,
            absolute_sigma=(sigma is not None),
            maxfev=maxfev,
        )
        A_fit, B_fit, Y0_fit = float(popt[0]), float(popt[1]), float(popt[2])
        perr = np.sqrt(np.maximum(0.0, np.diag(pcov)))
        A_err, B_err, Y0_err = float(perr[0]), float(perr[1]), float(perr[2])
    except RuntimeError:
        A_fit, B_fit, Y0_fit = A0, B0, Y0_0
        A_err = B_err = Y0_err = float("nan")
        converged = False

    y_fit = _model(x, A_fit, B_fit, Y0_fit)
    r2 = _r2(y, y_fit)
    residuals = y - y_fit
    if sigma is not None:
        chi_sqr = float(np.sum((residuals / np.asarray(sigma, dtype=float)) ** 2))
    else:
        chi_sqr = float(np.sum(residuals ** 2)) / n

    return {
        "A":         A_fit,
        "Tau":       B_fit,
        "X0":        float(x_origin),
        "Y0":        Y0_fit,
        "A_err":     A_err,
        "Tau_err":   B_err,
        "Y0_err":    Y0_err,
        "r2":        r2,
        "chi_sqr":   chi_sqr,
        "yfit":      y_fit,
        "residuals": residuals,
        "x":         x,
        "n":         n,
        "converged": converged,
    }


def fit_gauss(
    yarray: np.ndarray,
    xstart: float = 0.0,
    xdelta: float = 1.0,
    xarray: np.ndarray | None = None,
    xbgn: float = -math.inf,
    xend: float = math.inf,
    p0: dict | None = None,
    sigma: np.ndarray | None = None,
    maxfev: int = 10000,
    ignore_nans: bool = True,
) -> dict:
    """Fit a Gaussian to *yarray*.

    Model: ``f(x) = A * exp(-0.5 * ((x - mu) / sigma)^2) + Y0``

    Missing initial-parameter keys are auto-estimated:
    ``A = y at peak (with sign)``, ``mu = x at peak``,
    ``sigma = (x[-1] - x[0]) / 4``, ``Y0 = mean(y)``.

    Args:
        yarray:  1-D numpy array of y-values.
        xstart:  X-axis start value (uniform spacing).
        xdelta:  X-axis sample interval (uniform spacing).
        xarray:  Non-uniform x-values array.
        xbgn:    Fit window start (x-units). Default ``-inf``.
        xend:    Fit window end (x-units). Default ``+inf``.
        p0:      Initial parameter estimates with optional keys ``"A"``,
                 ``"Mu"``, ``"Sigma"``, ``"Y0"``.
        sigma:   Per-point standard deviations for weighted fitting.
        ignore_nans: If True (default), exclude NaN values before fitting.

    Returns:
        Dict with keys ``"A"``, ``"Mu"``, ``"Sigma"``, ``"Y0"``, ``"r2"``,
        ``"chi_sqr"``, ``"n"``, ``"converged"`` (bool).

    Raises:
        ValueError: Fewer than 4 data points, or *sigma* length mismatch.
    """
    from scipy.optimize import curve_fit  # noqa: PLC0415

    x, y = _extract_xy_window(yarray, xstart, xdelta, xarray, xbgn, xend, ignore_nans)
    n = len(x)
    if n < 4:
        raise ValueError("fit_gauss: need at least 4 data points, got %d" % n)

    if sigma is not None and len(sigma) != n:
        raise ValueError(
            "fit_gauss: sigma length (%d) != windowed data length (%d)" % (len(sigma), n)
        )

    p0_dict = p0 or {}
    i_peak = int(np.argmax(np.abs(y - np.mean(y))))
    A0     = p0_dict.get("A",     float(y[i_peak]))
    mu0    = p0_dict.get("Mu",    float(x[i_peak]))
    x_span = float(x[-1] - x[0])
    sigma0 = p0_dict.get("Sigma", x_span / 4.0 if x_span > 0 else 1.0)
    Y0_0   = p0_dict.get("Y0",   float(np.mean(y)))

    def _model(xv: np.ndarray, A: float, mu: float, sg: float, Y0: float) -> np.ndarray:
        return A * np.exp(-0.5 * ((xv - mu) / sg) ** 2) + Y0

    converged = True
    try:
        popt, pcov = curve_fit(
            _model, x, y,
            p0=[A0, mu0, sigma0, Y0_0],
            sigma=sigma,
            absolute_sigma=(sigma is not None),
            maxfev=maxfev,
        )
        A_fit   = float(popt[0])
        mu_fit  = float(popt[1])
        sg_fit  = float(popt[2])
        Y0_fit  = float(popt[3])
        perr = np.sqrt(np.maximum(0.0, np.diag(pcov)))
        A_err, mu_err, sg_err, Y0_err = (
            float(perr[0]), float(perr[1]), float(perr[2]), float(perr[3])
        )
    except RuntimeError:
        A_fit, mu_fit, sg_fit, Y0_fit = A0, mu0, sigma0, Y0_0
        A_err = mu_err = sg_err = Y0_err = float("nan")
        converged = False

    y_fit = _model(x, A_fit, mu_fit, sg_fit, Y0_fit)
    r2 = _r2(y, y_fit)
    residuals = y - y_fit
    if sigma is not None:
        chi_sqr = float(np.sum((residuals / np.asarray(sigma, dtype=float)) ** 2))
    else:
        chi_sqr = float(np.sum(residuals ** 2)) / n

    return {
        "A":         A_fit,
        "Mu":        mu_fit,
        "Sigma":     sg_fit,
        "Y0":        Y0_fit,
        "A_err":     A_err,
        "Mu_err":    mu_err,
        "Sigma_err": sg_err,
        "Y0_err":    Y0_err,
        "r2":        r2,
        "chi_sqr":   chi_sqr,
        "yfit":      y_fit,
        "residuals": residuals,
        "x":         x,
        "n":         n,
        "converged": converged,
    }


def fit_exp2(
    yarray: np.ndarray,
    xstart: float = 0.0,
    xdelta: float = 1.0,
    xarray: np.ndarray | None = None,
    xbgn: float = -math.inf,
    xend: float = math.inf,
    x_origin: float = 0.0,
    p0: dict | None = None,
    sigma: np.ndarray | None = None,
    maxfev: int = 10000,
    ignore_nans: bool = True,
) -> dict:
    """Fit a double exponential to *yarray*.

    Model: ``f(x) = A1 * exp(-(x - X0) / Tau1) + A2 * exp(-(x - X0) / Tau2) + Y0``

    where:

    * **A1**, **A2** — amplitudes
    * **Tau1**, **Tau2** — time constants (sorted so Tau1 ≤ Tau2 after fitting)
    * **X0** — fixed x-origin (``x_origin``); not fitted
    * **Y0** — y-offset

    Missing initial-parameter keys in *p0* are auto-estimated:
    ``A1 = A2 = (y[0] - y[-1]) / 2``, ``Tau1 = range/5``,
    ``Tau2 = range/2``, ``Y0 = y[-1]``.

    Args:
        yarray:    1-D numpy array of y-values.
        xstart:    X-axis start value (uniform spacing).
        xdelta:    X-axis sample interval (uniform spacing).
        xarray:    Non-uniform x-values array; overrides *xstart*/*xdelta*.
        xbgn:      Fit window start (x-units). Default ``-inf``.
        xend:      Fit window end (x-units). Default ``+inf``.
        x_origin:  Fixed x-offset (X0) in the model. Default 0.0.
        p0:        Initial parameter estimates as a dict with optional keys
                   ``"A1"``, ``"Tau1"``, ``"A2"``, ``"Tau2"``, ``"Y0"``.
                   Missing keys are auto-estimated.
        sigma:     Per-point standard deviations for weighted fitting.
        maxfev:    Maximum function evaluations for the optimizer. Default 10000.
        ignore_nans: If True (default), exclude NaN values before fitting.

    Returns:
        Dict with keys:

        * ``"A1"``, ``"Tau1"``, ``"A2"``, ``"Tau2"``, ``"X0"``, ``"Y0"``
          — fitted parameters (Tau1 ≤ Tau2; ``"X0"`` echoes ``x_origin``)
        * ``"A1_err"``, ``"Tau1_err"``, ``"A2_err"``, ``"Tau2_err"``,
          ``"Y0_err"`` — one-standard-deviation parameter uncertainties
        * ``"r2"``, ``"chi_sqr"``, ``"yfit"``, ``"residuals"``, ``"x"``,
          ``"n"``, ``"converged"``

    Raises:
        ValueError: Fewer than 5 data points, or *sigma* length mismatch.
    """
    from scipy.optimize import curve_fit  # noqa: PLC0415

    x, y = _extract_xy_window(yarray, xstart, xdelta, xarray, xbgn, xend, ignore_nans)
    n = len(x)
    if n < 5:
        raise ValueError("fit_exp2: need at least 5 data points, got %d" % n)

    if sigma is not None and len(sigma) != n:
        raise ValueError(
            "fit_exp2: sigma length (%d) != windowed data length (%d)" % (len(sigma), n)
        )

    p0_dict = p0 or {}
    _win = max(1, n // 10)
    _y_start = float(np.nanmedian(y[:_win]))
    _y_end   = float(np.nanmedian(y[-_win:]))
    _half_amp = (_y_start - _y_end) / 2.0
    x_span = float(x[-1] - x[0])
    A1_0   = p0_dict.get("A1",   _half_amp)
    Tau1_0 = p0_dict.get("Tau1", x_span / 5.0 if x_span > 0 else 1.0)
    A2_0   = p0_dict.get("A2",   _half_amp)
    Tau2_0 = p0_dict.get("Tau2", x_span / 2.0 if x_span > 0 else 2.0)
    Y0_0   = p0_dict.get("Y0",   _y_end)

    def _model(xv, A1, B1, A2, B2, Y0):
        return A1 * np.exp(-(xv - x_origin) / B1) + A2 * np.exp(-(xv - x_origin) / B2) + Y0

    converged = True
    try:
        popt, pcov = curve_fit(
            _model, x, y,
            p0=[A1_0, Tau1_0, A2_0, Tau2_0, Y0_0],
            sigma=sigma,
            absolute_sigma=(sigma is not None),
            maxfev=maxfev,
        )
        A1_f, B1_f, A2_f, B2_f, Y0_f = [float(v) for v in popt]
        perr = np.sqrt(np.maximum(0.0, np.diag(pcov)))
        A1_e, B1_e, A2_e, B2_e, Y0_e = [float(v) for v in perr]
        # Sort so Tau1 <= Tau2
        if B1_f > B2_f:
            A1_f, B1_f, A1_e, B1_e, A2_f, B2_f, A2_e, B2_e = (
                A2_f, B2_f, A2_e, B2_e, A1_f, B1_f, A1_e, B1_e
            )
    except RuntimeError:
        A1_f, B1_f, A2_f, B2_f, Y0_f = A1_0, Tau1_0, A2_0, Tau2_0, Y0_0
        A1_e = B1_e = A2_e = B2_e = Y0_e = float("nan")
        converged = False

    y_fit = _model(x, A1_f, B1_f, A2_f, B2_f, Y0_f)
    r2 = _r2(y, y_fit)
    residuals = y - y_fit
    if sigma is not None:
        chi_sqr = float(np.sum((residuals / np.asarray(sigma, dtype=float)) ** 2))
    else:
        chi_sqr = float(np.sum(residuals ** 2)) / n

    return {
        "A1":      A1_f,
        "Tau1":    B1_f,
        "A2":      A2_f,
        "Tau2":    B2_f,
        "X0":      float(x_origin),
        "Y0":      Y0_f,
        "A1_err":  A1_e,
        "Tau1_err": B1_e,
        "A2_err":  A2_e,
        "Tau2_err": B2_e,
        "Y0_err":  Y0_e,
        "r2":        r2,
        "chi_sqr":   chi_sqr,
        "yfit":      y_fit,
        "residuals": residuals,
        "x":         x,
        "n":         n,
        "converged": converged,
    }


def fit_boltzmann(
    yarray: np.ndarray,
    xstart: float = 0.0,
    xdelta: float = 1.0,
    xarray: np.ndarray | None = None,
    xbgn: float = -math.inf,
    xend: float = math.inf,
    p0: dict | None = None,
    sigma: np.ndarray | None = None,
    maxfev: int = 10000,
    ignore_nans: bool = True,
) -> dict:
    """Fit a Boltzmann sigmoid to *yarray*.

    Model: ``f(x) = A / (1 + exp(-(x - X50) / K)) + Y0``

    where:

    * **A** — amplitude (range of the sigmoid)
    * **X50** — midpoint (x at half-maximum)
    * **K** — slope factor (positive = rising, negative = falling)
    * **Y0** — y-offset (baseline)

    Missing initial-parameter keys in *p0* are auto-estimated:
    ``Y0 = min(y)``, ``A = max(y) - min(y)``,
    ``X50 = x at y closest to Y0 + A/2``, ``K = (x[-1] - x[0]) / 10``.

    Args:
        yarray:    1-D numpy array of y-values.
        xstart:    X-axis start value (uniform spacing).
        xdelta:    X-axis sample interval (uniform spacing).
        xarray:    Non-uniform x-values array; overrides *xstart*/*xdelta*.
        xbgn:      Fit window start (x-units). Default ``-inf``.
        xend:      Fit window end (x-units). Default ``+inf``.
        p0:        Initial parameter estimates as a dict with optional keys
                   ``"A"``, ``"X50"``, ``"K"``, ``"Y0"``.
                   Missing keys are auto-estimated.
        sigma:     Per-point standard deviations for weighted fitting.
        maxfev:    Maximum function evaluations for the optimizer. Default 10000.
        ignore_nans: If True (default), exclude NaN values before fitting.

    Returns:
        Dict with keys:

        * ``"A"``, ``"X50"``, ``"K"``, ``"Y0"`` — fitted parameters
        * ``"A_err"``, ``"X50_err"``, ``"K_err"``, ``"Y0_err"``
          — one-standard-deviation parameter uncertainties
        * ``"r2"``, ``"chi_sqr"``, ``"yfit"``, ``"residuals"``, ``"x"``,
          ``"n"``, ``"converged"``

    Raises:
        ValueError: Fewer than 4 data points, or *sigma* length mismatch.
    """
    from scipy.optimize import curve_fit  # noqa: PLC0415

    x, y = _extract_xy_window(yarray, xstart, xdelta, xarray, xbgn, xend, ignore_nans)
    n = len(x)
    if n < 4:
        raise ValueError("fit_boltzmann: need at least 4 data points, got %d" % n)

    if sigma is not None and len(sigma) != n:
        raise ValueError(
            "fit_boltzmann: sigma length (%d) != windowed data length (%d)"
            % (len(sigma), n)
        )

    p0_dict = p0 or {}
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    x_span = float(x[-1] - x[0])
    Y0_0  = p0_dict.get("Y0",  y_min)
    A0    = p0_dict.get("A",   y_max - y_min)
    # x closest to half-maximum
    half  = Y0_0 + A0 / 2.0
    X50_0 = p0_dict.get("X50", float(x[int(np.argmin(np.abs(y - half)))]))
    # Sign of K: positive for rising, negative for falling sigmoid
    _K_mag = x_span / 10.0 if x_span > 0 else 1.0
    _rising = float(np.nanmean(y[n // 2:])) >= float(np.nanmean(y[:n // 2]))
    K0    = p0_dict.get("K",   _K_mag if _rising else -_K_mag)

    def _model(xv, A, X50, K, Y0):
        return A / (1.0 + np.exp(-(xv - X50) / K)) + Y0

    converged = True
    try:
        popt, pcov = curve_fit(
            _model, x, y,
            p0=[A0, X50_0, K0, Y0_0],
            sigma=sigma,
            absolute_sigma=(sigma is not None),
            maxfev=maxfev,
        )
        A_f, X50_f, K_f, Y0_f = [float(v) for v in popt]
        perr = np.sqrt(np.maximum(0.0, np.diag(pcov)))
        A_e, X50_e, K_e, Y0_e = [float(v) for v in perr]
    except RuntimeError:
        A_f, X50_f, K_f, Y0_f = A0, X50_0, K0, Y0_0
        A_e = X50_e = K_e = Y0_e = float("nan")
        converged = False

    y_fit = _model(x, A_f, X50_f, K_f, Y0_f)
    r2 = _r2(y, y_fit)
    residuals = y - y_fit
    if sigma is not None:
        chi_sqr = float(np.sum((residuals / np.asarray(sigma, dtype=float)) ** 2))
    else:
        chi_sqr = float(np.sum(residuals ** 2)) / n

    return {
        "A":       A_f,
        "X50":     X50_f,
        "K":       K_f,
        "Y0":      Y0_f,
        "A_err":   A_e,
        "X50_err": X50_e,
        "K_err":   K_e,
        "Y0_err":  Y0_e,
        "r2":        r2,
        "chi_sqr":   chi_sqr,
        "yfit":      y_fit,
        "residuals": residuals,
        "x":         x,
        "n":         n,
        "converged": converged,
    }
