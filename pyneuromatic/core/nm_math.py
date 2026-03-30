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
# Time-window helper
# =========================================================================


def time_window_to_slice(
    arr: np.ndarray,
    xscale_dict: dict,
    x0: float,
    x1: float,
) -> slice:
    """Convert a time window to an array slice using xscale start/delta.

    Clips to valid range; returns an empty slice if the window is fully
    outside the array bounds.  Infinite bounds are treated as array
    boundaries (``-inf`` → index 0, ``+inf`` → index len(arr)).

    Args:
        arr:         Array whose length defines the valid index range.
        xscale_dict: Dict with ``"start"`` and ``"delta"`` keys (floats).
        x0:     Start of the time window.  ``-inf`` selects from the
                     beginning of the array.
        x1:       End of the time window (inclusive).  ``+inf`` selects
                     to the end of the array.

    Returns:
        A :class:`slice` suitable for indexing *arr*.
    """
    start = xscale_dict.get("start", 0.0)
    delta = xscale_dict.get("delta", 1.0)
    if delta == 0:
        return slice(0, 0)
    i0 = 0 if math.isinf(x0) else int(round((x0 - start) / delta))
    i1 = len(arr) if math.isinf(x1) else int(round((x1 - start) / delta)) + 1
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

    Fits a line through (x0, y0) and (x1, y1) and returns the x-value
    where that line equals *ylevel*.

    Args:
        ylevel: The y-axis threshold to intersect.
        x0, y0: Coordinates of the first point.
        x1, y1: Coordinates of the second point.

    Returns:
        Interpolated x-value (float) at y = ylevel.
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
    i_nearest: bool = True,
    x_interp: bool = True,
    ignore_nans: bool = True,
) -> tuple:
    """Find crossings of a y-axis level in a data array.

    A crossing is detected wherever the signal transitions across *ylevel*
    (i.e. ``np.diff(yarray > ylevel)`` is True). For each crossing, the
    nearest sample index and interpolated x-value are returned.

    Args:
        yarray:      1-D numpy array of y-values to search.
        ylevel:      The y-axis threshold to search for.
        func_name:   Controls which crossing directions are returned:
                     ``"level"`` — all crossings (both rising and falling).
                     ``"level+"`` — rising crossings only.
                     ``"level-"`` — falling crossings only.
        xarray:      Optional 1-D numpy array of x-values (same size as
                     *yarray*). Used instead of *xstart*/*xdelta* when
                     provided.
        xstart:      X-scale start value; used when *xarray* is None.
        xdelta:      X-scale sample interval; used when *xarray* is None.
        i_nearest:   If True (default), returns the sample index nearest to
                     the crossing via linear interpolation. If False, returns
                     the index immediately after the crossing.
        x_interp:    If True (default), returns the interpolated x-value at
                     the exact crossing. If False, returns the x-value at
                     the nearest sample index.
        ignore_nans: If True (default), NaN values are included in the
                     transition detection.

    Returns:
        Tuple ``(indexes, xvalues)`` of numpy arrays:
        *indexes* — sample indices (int) nearest to each crossing.
        *xvalues* — x-values (float) at each crossing location.
        Both arrays are empty if no crossings are found.

    Raises:
        TypeError:  If *func_name* is not a string, or *yarray*/*xarray* is
                    not a numpy ndarray.
        ValueError: If *func_name* does not contain ``"level"``, *ylevel*
                    is inf/nan, *xstart*/*xdelta* is inf/nan, or *xarray*
                    size differs from *yarray*.
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

    if not isinstance(i_nearest, bool):
        i_nearest = True
    if not isinstance(x_interp, bool):
        x_interp = True
    if not isinstance(ignore_nans, bool):
        ignore_nans = True

    level_crossings = np.diff(yarray > ylevel, prepend=False)
    locations = np.argwhere(level_crossings)
    if len(locations.shape) != 2:
        raise RuntimeError("locations shape should be 2")
    locations = locations[:, 0]
    indexes = []
    xvalues = []

    for i in locations:
        if i == 0:
            continue

        y0 = yarray[i - 1]
        y1 = yarray[i]

        if f == "level+":
            if y1 <= y0:
                continue
        elif f == "level-":
            if y1 >= y0:
                continue

        if found_xarray:
            x0 = xarray[i - 1]
            x1 = xarray[i]
            dx = x1 - x0
        else:
            x0 = xstart + (i - 1) * xdelta
            x1 = xstart + i * xdelta
            dx = xdelta

        if not i_nearest:
            indexes.append(i)
            xvalues.append(x1)
            continue

        dy = y1 - y0
        m = dy / dx
        b = y1 - m * x1
        x = (ylevel - b) / m

        if abs(x - x0) <= abs(x - x1):
            indexes.append(i - 1)
            xvalues.append(x if x_interp else x0)
        else:
            indexes.append(i)
            xvalues.append(x if x_interp else x1)

    return (np.array(indexes), np.array(xvalues))


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
        yarray:      1-D numpy array of y-values.
        xarray:      Optional 1-D numpy array of x-values (same size as
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

        x0 = xstart
        x1 = xstart + (yarray.size - 1) * xdelta
        xarray = np.linspace(x0, x1, yarray.size)

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
        y: 1-D numpy array of y-values.
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
        y: 1-D numpy array of y-values.
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
        y: 1-D numpy array of y-values.
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
# Stats functions
# =========================================================================


def histogram(
    y: np.ndarray,
    bins: int | list = 10,
    xrange: tuple | None = None,
    density: bool = False,
) -> dict:
    """Compute a histogram of a 1-D array, excluding NaN and Inf values.

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
    Silver (UCL) based on *Numerical Recipes*.

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
    trend over time using the Spearman rank-order correlation between values
    and their array indices. Searches from the largest possible window down to
    ``min_window``, stopping as soon as the largest stable window is found.

    This mirrors the Igor NeuroMatic ``NMStabilityRankOrderTest()`` function
    (``NM_StatsTabStability.ipf``, pass 1 only), originally written by
    Dr. Angus Silver and Simon Mitchell (UCL), based on *Numerical Recipes in C*.

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
