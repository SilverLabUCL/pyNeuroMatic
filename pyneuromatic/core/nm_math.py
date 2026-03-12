"""Pure math utility functions shared across pyNeuroMatic.

No NM object dependencies — only numpy.
"""
from __future__ import annotations

import numpy as np

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
