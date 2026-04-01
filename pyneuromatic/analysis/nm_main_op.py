# -*- coding: utf-8 -*-
"""
NMMainOp — operation classes for NMToolMain.

Provides a base class NMMainOp and concrete subclasses following the same
pattern as nm_transform.py:
one class per operation, a module-level registry, and a lookup helper.

Part of pyNeuroMatic, a Python implementation of NeuroMatic for analyzing,
acquiring and simulating electrophysiology data.

If you use this software in your research, please cite:
Rothman JS and Silver RA (2018) NeuroMatic: An Integrated Open-Source
Software Toolkit for Acquisition, Analysis and Simulation of
Electrophysiology Data. Front. Neuroinform. 12:14.
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
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_math as nm_math
import pyneuromatic.core.nm_utilities as nmu


def _extract_channels_epochs(
    data_items: list,
    prefix: str | None,
) -> tuple[list[str], list[int]]:
    """Extract sorted unique channel chars and epoch ints from data item names."""
    channels: set[str] = set()
    epochs: set[int] = set()
    for item, _ in data_items:
        parsed = nmu.parse_data_name(item.name)
        if parsed is not None:
            _, ch, ep = parsed
            channels.add(ch)
            epochs.add(ep)
    return sorted(channels), sorted(epochs)


def _epochs_repr(epochs: list[int]) -> str:
    """Compact representation of epoch list: range() for arithmetic sequences, else repr."""
    if len(epochs) < 3:
        return repr(epochs)
    step = epochs[1] - epochs[0]
    if all(b - a == step for a, b in zip(epochs, epochs[1:])):
        stop = epochs[-1] + step
        if step == 1:
            return "list(range(%d, %d))" % (epochs[0], stop)
        return "list(range(%d, %d, %d))" % (epochs[0], stop, step)
    return repr(epochs)


# =========================================================================
# Base class
# =========================================================================


class NMMainOp:
    """Base class for NMToolMain operations.

    Mirrors the NMTransform pattern: one subclass per operation, a
    module-level registry, and a ``run_all()`` primary interface.

    The default ``run_all()`` provides a ``run_init → run × N → run_finish``
    lifecycle.  Subclasses override the individual lifecycle methods:
    pointwise ops (e.g. Scale) override only ``run()``; aggregating ops
    (e.g. Average) also override ``run_init()`` and ``run_finish()``.

    Subclasses should set the class attribute ``name`` to a short lowercase
    string matching the registry key (e.g. ``"scale"``).

    Attributes:
        overwrite: If True (default), an existing output array with the same
            name is replaced in-place.  If False, a sequence number
            (``_0``, ``_1``, …) is appended to find a free name so previous
            results are preserved.
    """

    name: str = ""
    _overwrite: bool = True  # class-level default; overridden per-instance

    @property
    def overwrite(self) -> bool:
        """If True, replace an existing output array; if False, auto-sequence."""
        return self._overwrite

    @overwrite.setter
    def overwrite(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "overwrite", "bool"))
        self._overwrite = value

    def _make_out_name(self, folder: NMFolder, base_name: str) -> str:
        """Return the output array name, respecting the overwrite setting.

        If ``overwrite`` is True, returns *base_name* unchanged (existing
        array is replaced in-place).  If ``overwrite`` is False, always
        appends a sequence number (``_0``, ``_1``, …) so every run produces
        a new array and previous results are preserved.
        """
        if self._overwrite:
            return base_name
        i = 0
        while True:
            candidate = "%s_%d" % (base_name, i)
            if candidate not in folder.data:
                return candidate
            i += 1

    def _write_out_array(
        self,
        folder: NMFolder,
        out_name: str,
        nparray: np.ndarray,
        xscale: dict | None = None,
        yscale: dict | None = None,
    ) -> "NMData | None":
        """Create or overwrite an output array in *folder*.

        If an array named *out_name* already exists (overwrite=True case),
        its array and scales are updated in-place.  Otherwise a new NMData
        is created via ``folder.data.new()``.
        """
        existing = folder.data.get(out_name)
        if existing is not None:
            existing.nparray = nparray
            if xscale:
                xs = existing.xscale
                if "start" in xscale:
                    xs.start = xscale["start"]
                if "delta" in xscale:
                    xs.delta = xscale["delta"]
                if "label" in xscale:
                    xs.label = xscale["label"]
                if "units" in xscale:
                    xs.units = xscale["units"]
            if yscale:
                ys = existing.yscale
                if "label" in yscale:
                    ys.label = yscale["label"]
                if "units" in yscale:
                    ys.units = yscale["units"]
            return existing
        return folder.data.new(out_name, nparray=nparray,
                               xscale=xscale, yscale=yscale)

    def run_all(
        self,
        data_items: list[tuple[NMData, str | None]],
        folder: NMFolder | None,
        prefix: str | None = None,
    ) -> None:
        """Process all data items.

        Calls ``run_init()``, then ``run()`` for each item, then
        ``run_finish()``.  Available for standalone use (e.g. in tests);
        ``NMToolMain`` drives the lifecycle via its own ``run_init /
        run / run_finish`` hooks instead.

        Args:
            data_items: List of ``(NMData, channel_name)`` pairs.  The
                channel_name may be ``None`` when running in direct-data mode
                (no dataseries context).
            folder: The NMFolder that owns the source data.  Passed to
                ``run_finish()`` so ops can write output there.
            prefix: Dataseries name to use as the output array prefix.  If
                ``None``, ops fall back to parsing the prefix from the data
                name.

        Note:
            Before calling ``run_init()``, stores ``folder``, ``prefix``,
            item count, and the full item list as ``self._folder``,
            ``self._prefix``, ``self._n_items``, and ``self._data_items`` so
            that subclasses can access them in ``run_init()`` without
            overriding ``run_all()``.
        """
        self._folder = folder
        self._prefix = prefix
        self._n_items = len(data_items)
        self._data_items = data_items
        self.run_init()
        for data, channel_name in data_items:
            self.run(data, channel_name)
        self.run_finish(folder, prefix)

    def run_init(self) -> None:
        """Called once before the per-item loop.  Override to reset state."""

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Called for each data item.  Override for pointwise operations.

        Args:
            data: The NMData object to process.
            channel_name: Channel name from the selection context, or None.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError(
            "%s.run() not implemented" % self.__class__.__name__
        )

    def run_finish(
        self,
        folder: NMFolder | None = None,
        prefix: str | None = None,
    ) -> None:
        """Called once after the per-item loop.  Override to write results."""

    def _add_note(self, data: NMData, text: str) -> None:
        """Append a timestamped note to data.notes if available."""
        notes = getattr(data, "notes", None)
        if notes is not None:
            notes.add(text)

    def _add_op_note(self, data: NMData, body: str = "") -> None:
        """Append a note in the form ``OpName(body)`` to *data*.

        Equivalent to ``_add_note(data, f"{self._op_name}({body})")``.
        Use instead of :meth:`_add_note` whenever the note starts with the
        op name — avoids repeating ``self._op_name`` at every call site.
        """
        self._add_note(data, "%s(%s)" % (self._op_name, body))

    @property
    def _op_name(self) -> str:
        """Op name used in note strings, derived from the class name.

        ``NMMainOpBaseline`` → ``"NMBaseline"``, ``NMMainOpSum`` → ``"NMSum"``,
        etc.  Subclasses may override with a plain string attribute when the
        derived name is not suitable.
        """
        return type(self).__name__.replace("NMMainOp", "NM")

    def _op_params_str(self) -> str | None:
        """Return constructor keyword arguments as a string for this operation.

        Subclasses override to return a string of ``key=value`` pairs that
        exactly matches the op's constructor signature (using ``%r`` format so
        values are valid Python literals).  The returned string is shared by
        both :meth:`to_command_str` (for command history) and
        :meth:`run` / :meth:`run_finish` callers that build note strings.

        Return ``None`` to suppress command-history logging for this op.
        """
        return None

    def to_command_str(
        self,
        folder_name: str,
        prefix: str,
        channels: list[str],
        epochs: list[int],
    ) -> str | None:
        """Return a Python-executable command string for this operation.

        Delegates to :meth:`_op_params_str`; returns ``None`` when that
        method returns ``None`` (suppressing history logging).
        """
        params = self._op_params_str()
        if params is None:
            return None
        return self._base_cmd(params, folder_name, prefix, channels, epochs)

    def _base_cmd(
        self,
        op_kwargs: str,
        folder_name: str,
        prefix: str,
        channels: list[str],
        epochs: list[int],
    ) -> str:
        """Format a standard run_all() command string."""
        return (
            "%s(%s).run_all(\n    folder=%r, prefix=%r, channels=%r, epochs=%s)"
            % (type(self).__name__, op_kwargs, folder_name, prefix, channels, _epochs_repr(epochs))
        )


# =========================================================================
# Arithmetic (array-wise arithmetic operation)
# =========================================================================


class NMMainOpArithmetic(NMMainOp):
    """Apply an arithmetic operation to each selected data array.

    The ``factor`` parameter controls the operand:

    - **scalar** (``float``): the same factor is applied to every data item.
    - **list** (``list[float]``): factors consumed in order, one per data
      item.  List length must exactly match the number of data items.
    - **dict** (``dict[str, float]``): factor looked up by data name;
      missing keys are silently skipped.

    Parameters:
        factor: Operand — ``float``, ``list[float]``, or
            ``dict[str, float]``.  Default is ``1.0``.
        op: Operation string — one of ``"x"`` (multiply), ``"/"`` (divide),
            ``"+"`` (add), ``"-"`` (subtract), ``"="`` (assign),
            ``"**"`` (exponentiate).  Default is ``"x"``.
    """

    name = "arithmetic"

    def __init__(
        self,
        factor: float | list[float] | dict[str, float] = 1.0,
        op: str = "x",
    ) -> None:
        self.factor = factor
        self.op = op
        self._index: int = 0

    @property
    def factor(self) -> float | list[float] | dict[str, float]:
        """Operand — scalar, list, or dict."""
        return self._factor

    @factor.setter
    def factor(self, value: float | list[float] | dict[str, float]) -> None:
        if isinstance(value, bool):
            raise TypeError(
                nmu.type_error_str(value, "factor", "float, list, or dict")
            )
        if isinstance(value, (int, float)):
            self._factor = float(value)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise TypeError(
                        "factor[%d]: expected float, got %s" % (i, type(v).__name__)
                    )
            self._factor = value
        elif isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(k, str):
                    raise TypeError(
                        "factor key %r: expected str, got %s" % (k, type(k).__name__)
                    )
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise TypeError(
                        "factor[%r]: expected float, got %s" % (k, type(v).__name__)
                    )
            self._factor = value
        else:
            raise TypeError(
                "factor must be float, list, or dict, got %s" % type(value).__name__
            )

    @property
    def op(self) -> str:
        """Operation string."""
        return self._op

    @op.setter
    def op(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "op", "str"))
        if value not in nm_math.VALID_ARITH_OPS:
            raise ValueError(
                "op must be one of %s, got %r" % (sorted(nm_math.VALID_ARITH_OPS), value)
            )
        self._op = value

    def run_init(self) -> None:
        """Reset index and pre-validate list length before any run() calls."""
        self._index = 0
        if isinstance(self._factor, list) and len(self._factor) != self._n_items:
            raise IndexError(
                "NMMainOpArithmetic: factor list length must match number of "
                "data items (need %d, got %d)" % (self._n_items, len(self._factor))
            )

    def _get_factor(self, name: str) -> float | None:
        if isinstance(self._factor, (int, float)):
            return self._factor
        if isinstance(self._factor, list):
            f = float(self._factor[self._index])
            self._index += 1
            return f
        return self._factor.get(name)  # dict: None if name not found

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Apply factor and op to data.nparray in-place."""
        if not isinstance(data.nparray, np.ndarray):
            return
        factor = self._get_factor(data.name)
        if factor is None:
            return  # dict mode: name not in dict of factors — skip
        data.nparray = nm_math.apply_arithmetic(data.nparray.astype(float), factor, self._op)
        self._add_op_note(data, "factor=%.6g, op=%r" % (factor, self._op))

    def _op_params_str(self) -> str:
        return "factor=%r, op=%r" % (self._factor, self._op)


# =========================================================================
# Arithmetic (element-wise arithmetic operation)
# =========================================================================


class NMMainOpArithmeticByArray(NMMainOp):
    """Apply an element-wise arithmetic operation using a reference array.

    The operation is applied as ``data = data op ref`` element-by-element.
    When ``data`` and ``ref`` differ in length the operation is applied to
    the overlap (``min(len(data), len(ref))``); elements beyond the overlap
    are left unchanged.

    Parameters:
        ref: Reference operand — either an ``np.ndarray`` or a ``str`` data
            name that will be looked up in the source folder at runtime.
        op: Operation string — same choices as :class:`NMMainOpArithmetic`.
            Default is ``"x"``.
    """

    name = "arithmetic_by_array"

    def __init__(self, ref: np.ndarray | str | None = None, op: str = "x") -> None:
        self.ref = np.zeros(0) if ref is None else ref
        self.op = op
        self._resolved_ref: np.ndarray | None = None

    @property
    def ref(self) -> np.ndarray | str:
        """Reference operand (array or data name string)."""
        return self._ref

    @ref.setter
    def ref(self, value: np.ndarray | str) -> None:
        if isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "ref", "np.ndarray or str"))
        if not isinstance(value, (np.ndarray, str)):
            raise TypeError(nmu.type_error_str(value, "ref", "np.ndarray or str"))
        self._ref = value

    @property
    def op(self) -> str:
        """Operation string."""
        return self._op

    @op.setter
    def op(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "op", "str"))
        if value not in nm_math.VALID_ARITH_OPS:
            raise ValueError(
                "op must be one of %s, got %r" % (sorted(nm_math.VALID_ARITH_OPS), value)
            )
        self._op = value

    def run_init(self) -> None:
        """Resolve the reference array before the per-item loop.

        Reads ``self._folder`` (set by :meth:`NMMainOp.run_all`) to look up a
        string ref name.  Stores the resolved array in ``self._resolved_ref``.
        """
        folder = self._folder
        if isinstance(self._ref, str):
            if folder is None:
                raise ValueError(
                    "folder required to resolve ref %r" % self._ref
                )
            ref_data = folder.data.get(self._ref)
            if ref_data is None or not isinstance(ref_data.nparray, np.ndarray):
                raise ValueError(
                    "ref %r not found in folder %r" % (self._ref, folder.name)
                )
            self._resolved_ref = ref_data.nparray.astype(float)
        else:
            self._resolved_ref = self._ref.astype(float)
        ref_len = len(self._resolved_ref)
        for data, _ in self._data_items:
            if isinstance(data.nparray, np.ndarray) and len(data.nparray) != ref_len:
                raise ValueError(
                    "NMMainOpArithmeticByArray: length mismatch for %r "
                    "(data has %d points, ref has %d)"
                    % (data.name, len(data.nparray), ref_len)
                )

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Apply ref and op to data.nparray element-wise in-place."""
        if not isinstance(data.nparray, np.ndarray):
            return
        arr = data.nparray.astype(float)
        data.nparray = nm_math.apply_arithmetic(arr, self._resolved_ref, self._op)
        self._add_op_note(data, self._op_params_str())

    def _op_params_str(self) -> str:
        ref_repr = "%r" % self._ref if isinstance(self._ref, str) else "np.array([...])"
        return "ref=%s, op=%r" % (ref_repr, self._op)


# =========================================================================
# Redimension
# =========================================================================


class NMMainOpRedimension(NMMainOp):
    """Change the number of points in each selected array (in-place).

    Truncates when ``n_points`` < current length; pads with ``fill`` when
    ``n_points`` > current length.  Equivalent to Igor's ``Redimension/N=``.

    Parameters:
        n_points: New number of points (>= 1).  Default 0 means no change.
        fill: Value used to pad when extending (default 0.0).
    """

    name = "redimension"

    def __init__(self, n_points: int = 0, fill: float = 0.0) -> None:
        self.n_points = n_points  # setters for validation
        self.fill = fill

    @property
    def n_points(self) -> int:
        """New number of points (0 = no change)."""
        return self._n_points

    @n_points.setter
    def n_points(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "n_points", "int"))
        if value < 0:
            raise ValueError("n_points must be >= 0, got %d" % value)
        self._n_points = value

    @property
    def fill(self) -> float:
        """Pad value used when extending an array."""
        return self._fill

    @fill.setter
    def fill(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "fill", "float"))
        self._fill = float(value)

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Resize data.nparray to n_points in-place.

        Args:
            data: The NMData object to resize.
            channel_name: Unused; present for API consistency.
        """
        if not isinstance(data.nparray, np.ndarray) or self._n_points == 0:
            return
        arr = data.nparray
        n = self._n_points
        old_len = len(arr)
        if n <= old_len:
            data.nparray = arr[:n]
        else:
            data.nparray = np.concatenate([arr, np.full(n - old_len, self._fill)])
        self._add_op_note(data, self._op_params_str())

    def _op_params_str(self) -> str:
        return "n_points=%d, fill=%r" % (self._n_points, self._fill)


# =========================================================================
# Insert points
# =========================================================================


class NMMainOpInsertPoints(NMMainOp):
    """Insert points into each selected array at a given index (in-place).

    Points at and after ``index`` are shifted right.

    Parameters:
        index: Position at which to insert (0-based, default 0).
        n_points: Number of points to insert (default 1).
        fill: Value for the inserted points (default 0.0).
    """

    name = "insert_points"

    def __init__(self, index: int = 0, n_points: int = 1, fill: float = 0.0) -> None:
        self.index = index  # setters for validation
        self.n_points = n_points
        self.fill = fill

    @property
    def index(self) -> int:
        """Insertion position (0-based)."""
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "index", "int"))
        if value < 0:
            raise ValueError("index must be >= 0, got %d" % value)
        self._index = value

    @property
    def n_points(self) -> int:
        """Number of points to insert."""
        return self._n_points

    @n_points.setter
    def n_points(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "n_points", "int"))
        if value < 1:
            raise ValueError("n_points must be >= 1, got %d" % value)
        self._n_points = value

    @property
    def fill(self) -> float:
        """Value assigned to the inserted points."""
        return self._fill

    @fill.setter
    def fill(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "fill", "float"))
        self._fill = float(value)

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Insert n_points at index in data.nparray in-place.

        Args:
            data: The NMData object to modify.
            channel_name: Unused; present for API consistency.
        """
        if not isinstance(data.nparray, np.ndarray):
            return
        data.nparray = np.insert(
            data.nparray, self._index, np.full(self._n_points, self._fill)
        )
        self._add_op_note(data, self._op_params_str())

    def _op_params_str(self) -> str:
        return "index=%d, n_points=%d, fill=%r" % (
            self._index, self._n_points, self._fill)


# =========================================================================
# Delete points
# =========================================================================


class NMMainOpDeletePoints(NMMainOp):
    """Delete points from each selected array at a given index (in-place).

    Parameters:
        index: Position of the first point to delete (0-based, default 0).
        n_points: Number of points to delete (default 1).
    """

    name = "delete_points"

    def __init__(self, index: int = 0, n_points: int = 1) -> None:
        self.index = index  # setters for validation
        self.n_points = n_points

    @property
    def index(self) -> int:
        """Position of the first point to delete (0-based)."""
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "index", "int"))
        if value < 0:
            raise ValueError("index must be >= 0, got %d" % value)
        self._index = value

    @property
    def n_points(self) -> int:
        """Number of points to delete."""
        return self._n_points

    @n_points.setter
    def n_points(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "n_points", "int"))
        if value < 1:
            raise ValueError("n_points must be >= 1, got %d" % value)
        self._n_points = value

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Delete n_points starting at index from data.nparray in-place.

        Args:
            data: The NMData object to modify.
            channel_name: Unused; present for API consistency.
        """
        if not isinstance(data.nparray, np.ndarray):
            return
        if self._index >= len(data.nparray):
            return  # nothing to delete
        data.nparray = np.delete(
            data.nparray, np.arange(self._index, self._index + self._n_points)
        )
        self._add_op_note(data, self._op_params_str())

    def _op_params_str(self) -> str:
        return "index=%d, n_points=%d" % (self._index, self._n_points)


# =========================================================================
# Reverse
# =========================================================================


class NMMainOpReverse(NMMainOp):
    """Reverse each selected array in-place.

    Equivalent to ``np.flip(arr)``.  No parameters.
    """

    name = "reverse"

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Reverse data.nparray in-place.

        Args:
            data: The NMData object to reverse.
            channel_name: Unused; present for API consistency.
        """
        if not isinstance(data.nparray, np.ndarray):
            return
        data.nparray = np.flip(data.nparray)
        self._add_op_note(data)

    def _op_params_str(self) -> str:
        return ""


# =========================================================================
# Rotate
# =========================================================================


class NMMainOpRotate(NMMainOp):
    """Rotate each selected array by ``n_points`` positions (in-place).

    Uses ``np.roll(arr, n_points)``.  Positive values shift elements to
    the right (last element wraps to the front); negative values shift
    left.

    Parameters:
        n_points: Number of positions to rotate (default 1).  May be
            negative.  Must be int (not bool).
    """

    name = "rotate"

    def __init__(self, n_points: int = 1) -> None:
        self.n_points = n_points  # setter for validation

    @property
    def n_points(self) -> int:
        """Number of positions to rotate (negative = rotate left)."""
        return self._n_points

    @n_points.setter
    def n_points(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "n_points", "int"))
        self._n_points = value

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Rotate data.nparray by n_points positions in-place.

        Args:
            data: The NMData object to rotate.
            channel_name: Unused; present for API consistency.
        """
        if not isinstance(data.nparray, np.ndarray):
            return
        data.nparray = np.roll(data.nparray, self._n_points)
        self._add_op_note(data, self._op_params_str())

    def _op_params_str(self) -> str:
        return "n_points=%d" % self._n_points


# =========================================================================
# Integrate
# =========================================================================


class NMMainOpIntegrate(NMMainOp):
    """Cumulative integration of each selected array (in-place).

    Two methods are supported:

    - **rectangular**: ``np.cumsum(arr) * delta`` — equivalent to summing
      rectangular strips of width ``delta``.
    - **trapezoid**: cumulative trapezoidal rule — each step area is
      ``0.5 * (y[i] + y[i+1]) * delta``; the first output point is 0.0
      so the output length equals the input length.

    Parameters:
        method: ``"rectangular"`` (default) or ``"trapezoid"``.
    """

    name = "integrate"

    _VALID_METHODS = {"rectangular", "trapezoid"}

    def __init__(self, method: str = "rectangular") -> None:
        self.method = method  # setter for validation

    @property
    def method(self) -> str:
        """Integration method: ``'rectangular'`` or ``'trapezoid'``."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "method", "string"))
        if value not in self._VALID_METHODS:
            raise ValueError(
                "method must be one of %s, got %r" % (sorted(self._VALID_METHODS), value)
            )
        self._method = value

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Integrate data.nparray in-place.

        Args:
            data: The NMData object to integrate.
            channel_name: Unused; present for API consistency.
        """
        if not isinstance(data.nparray, np.ndarray):
            return
        delta = data.xscale.delta
        arr = data.nparray.astype(float)
        if self._method == "rectangular":
            data.nparray = np.cumsum(arr) * delta
        else:  # "trapezoid"
            steps = 0.5 * (arr[:-1] + arr[1:]) * delta
            data.nparray = np.concatenate([[0.0], np.cumsum(steps)])
        self._add_op_note(data, self._op_params_str())

    def _op_params_str(self) -> str:
        return "method=%r" % self._method


# =========================================================================
# Differentiate
# =========================================================================


class NMMainOpDifferentiate(NMMainOp):
    """First derivative of each selected array using ``np.gradient`` (in-place).

    Uses central differences for interior points and one-sided differences at
    the boundaries.  Preserves array length.  Scales by ``xscale.delta`` so
    the result has correct dy/dx units.  No parameters.
    """

    name = "differentiate"

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Differentiate data.nparray in-place.

        Args:
            data: The NMData object to differentiate.
            channel_name: Unused; present for API consistency.
        """
        if not isinstance(data.nparray, np.ndarray):
            return
        delta = data.xscale.delta
        arr = data.nparray.astype(float)
        if delta != 0:
            data.nparray = np.gradient(arr, delta)
        else:
            data.nparray = np.gradient(arr)
        self._add_op_note(data)

    def _op_params_str(self) -> str:
        return ""


# =========================================================================
# Replace values
# =========================================================================


class NMMainOpReplaceValues(NMMainOp):
    """Replace all points equal to ``old_value`` with ``new_value`` (in-place).

    NaN-aware: if ``old_value`` is NaN, ``np.isnan()`` is used to build the
    mask (since ``nan != nan``).  The note reports how many points were
    replaced; it is written even when no replacements occurred (n=0).

    Parameters:
        old_value: Value to search for (default 0.0).  ``float("nan")`` and
            ``float("inf")`` are accepted.
        new_value: Replacement value (default 0.0).
    """

    name = "replace_values"

    def __init__(self, old_value: float = 0.0, new_value: float = 0.0) -> None:
        self.old_value = old_value  # setters for validation
        self.new_value = new_value

    @property
    def old_value(self) -> float:
        """Value to search for."""
        return self._old_value

    @old_value.setter
    def old_value(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "old_value", "float"))
        self._old_value = float(value)

    @property
    def new_value(self) -> float:
        """Replacement value."""
        return self._new_value

    @new_value.setter
    def new_value(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "new_value", "float"))
        self._new_value = float(value)

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Replace matching values in data.nparray in-place.

        Args:
            data: The NMData object to modify.
            channel_name: Unused; present for API consistency.
        """
        if not isinstance(data.nparray, np.ndarray):
            return
        import math
        arr = data.nparray.astype(float)
        if math.isnan(self._old_value):
            mask = np.isnan(arr)
        else:
            mask = arr == self._old_value
        n = int(np.count_nonzero(mask))
        arr[mask] = self._new_value
        data.nparray = arr
        self._add_op_note(data, "%s, n=%d" % (self._op_params_str(), n))

    def _op_params_str(self) -> str:
        return "old_value=%r, new_value=%r" % (self._old_value, self._new_value)


# =========================================================================
# Delete NaNs
# =========================================================================


class NMMainOpDeleteNaNs(NMMainOp):
    """Remove NaN and/or ±Inf points from each selected array (in-place).

    Shortens the array.  The note reports how many points were removed;
    it is written even when n=0.

    Parameters:
        delete_nans: If True (default), remove NaN points.
        delete_infs: If True, remove ±Inf points (default False).

    At least one of ``delete_nans`` or ``delete_infs`` must be True.
    """

    name = "delete_nans"

    def __init__(self, delete_nans: bool = True, delete_infs: bool = False) -> None:
        self.delete_nans = delete_nans  # setters for validation
        self.delete_infs = delete_infs
        if not self._delete_nans and not self._delete_infs:
            raise ValueError(
                "at least one of delete_nans or delete_infs must be True"
            )

    @property
    def delete_nans(self) -> bool:
        """If True, NaN points are removed."""
        return self._delete_nans

    @delete_nans.setter
    def delete_nans(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "delete_nans", "boolean"))
        self._delete_nans = value

    @property
    def delete_infs(self) -> bool:
        """If True, ±Inf points are removed (default False)."""
        return self._delete_infs

    @delete_infs.setter
    def delete_infs(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "delete_infs", "boolean"))
        self._delete_infs = value

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Remove NaN/Inf points from data.nparray in-place.

        Args:
            data: The NMData object to modify.
            channel_name: Unused; present for API consistency.
        """
        if not isinstance(data.nparray, np.ndarray):
            return
        arr = data.nparray.astype(float)
        mask = np.zeros(len(arr), dtype=bool)
        if self._delete_nans:
            mask |= np.isnan(arr)
        if self._delete_infs:
            mask |= np.isinf(arr)
        n = int(np.count_nonzero(mask))
        data.nparray = arr[~mask]
        self._add_op_note(data, "%s, n=%d" % (self._op_params_str(), n))

    def _op_params_str(self) -> str:
        return "delete_nans=%r, delete_infs=%r" % (self._delete_nans, self._delete_infs)


# =========================================================================
# Baseline
# =========================================================================


class NMMainOpBaseline(NMMainOp):
    """Subtract a baseline from each selected array.

    Two modes are supported:

    - **per_array**: Each array's own baseline (mean of the window) is subtracted
      from that array independently.
    - **average**: A single shared baseline per channel is computed as the mean
      of all per-array baselines for that channel, then subtracted from every
      array in that channel.

    Parameters:
        x0: Baseline window start in xscale units (default 0.0).
        x1: Baseline window end in xscale units (default 0.0).  Must be >=
            ``x0``.
        mode: ``"per_array"`` (default) or ``"average"``.
        ignore_nans: If True (default) use ``np.nanmean``; otherwise ``np.mean``
            (NaN propagates to the result).
    """

    name = "baseline"

    _VALID_MODES = {"per_array", "average"}

    def __init__(
        self,
        x0: float = 0.0,
        x1: float = 0.0,
        mode: str = "per_array",
        ignore_nans: bool = True,
    ) -> None:
        self.x0 = x0
        self.x1 = x1
        self.mode = mode
        self.ignore_nans = ignore_nans

    # ------------------------------------------------------------------
    # Properties

    @property
    def x0(self) -> float:
        """Baseline window start (xscale units)."""
        return self._x0

    @x0.setter
    def x0(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x0", "float"))
        if math.isnan(float(value)):
            raise ValueError("x0 must not be NaN")
        self._x0 = float(value)

    @property
    def x1(self) -> float:
        """Baseline window end (xscale units)."""
        return self._x1

    @x1.setter
    def x1(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x1", "float"))
        if math.isnan(float(value)):
            raise ValueError("x1 must not be NaN")
        self._x1 = float(value)

    @property
    def mode(self) -> str:
        """Subtraction mode: ``'per_array'`` or ``'average'``."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "mode", "string"))
        if value not in self._VALID_MODES:
            raise ValueError(
                "mode must be one of %s, got %r" % (sorted(self._VALID_MODES), value)
            )
        self._mode = value

    @property
    def ignore_nans(self) -> bool:
        """If True, NaN values are excluded from baseline mean (np.nanmean)."""
        return self._ignore_nans

    @ignore_nans.setter
    def ignore_nans(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "ignore_nans", "boolean"))
        self._ignore_nans = value

    # ------------------------------------------------------------------
    # Validation helper

    def _validate_window(self) -> None:
        if self._x1 < self._x0:
            raise ValueError(
                "x1 (%g) must be >= x0 (%g)" % (self._x1, self._x0)
            )

    # ------------------------------------------------------------------
    # Lifecycle

    def run_init(self) -> None:
        """Reset per-run accumulators."""
        self._validate_window()
        self._baseline_accum: dict[str, list[float]] = {}
        self._data_refs: dict[str, list[NMData]] = {}

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Compute and (optionally) apply baseline for one array.

        Args:
            data: The NMData object to process.
            channel_name: Channel name from the selection context, or None
                (parsed from data.name as a fallback).
        """
        if not isinstance(data.nparray, np.ndarray):
            return

        if channel_name is None:
            parsed = nmu.parse_data_name(data.name)
            channel_name = parsed[1] if parsed is not None else "A"

        sl = nm_math.xscale_window_to_slice(
            data.nparray, data.xscale.to_dict(), self._x0, self._x1
        )
        segment = data.nparray[sl].astype(float)
        if len(segment) == 0:
            baseline = 0.0
        elif self._ignore_nans:
            baseline = float(np.nanmean(segment))
        else:
            baseline = float(np.mean(segment))

        if self._mode == "per_array":
            data.nparray = data.nparray.astype(float) - baseline
            self._add_op_note(data, "%s, baseline=%.6g" % (self._op_params_str(), baseline))
        else:  # "average"
            self._baseline_accum.setdefault(channel_name, []).append(baseline)
            self._data_refs.setdefault(channel_name, []).append(data)

    def run_finish(
        self,
        folder: NMFolder | None = None,
        prefix: str | None = None,
    ) -> None:
        """Apply averaged baseline (average mode only).

        In ``per_array`` mode this is a no-op (subtraction was done in ``run()``).
        In ``average`` mode the average of all per-array baselines for each channel
        is computed and subtracted from every array in that channel.
        """
        if self._mode == "per_array":
            return
        for channel_name, baselines in self._baseline_accum.items():
            avg_baseline = float(
                np.nanmean(baselines) if self._ignore_nans else np.mean(baselines)
            )
            for d in self._data_refs[channel_name]:
                d.nparray = d.nparray.astype(float) - avg_baseline
                self._add_op_note(
                    d, "%s, channel=%s, baseline=%.6g"
                    % (self._op_params_str(), channel_name, avg_baseline),
                )

    def _op_params_str(self) -> str:
        return "x0=%r, x1=%r, mode=%r, ignore_nans=%r" % (
            self._x0, self._x1, self._mode, self._ignore_nans)


# =========================================================================
# Smooth
# =========================================================================

_VALID_SMOOTH_METHODS: frozenset[str] = frozenset({"boxcar", "binomial", "savgol"})


class NMMainOpSmooth(NMMainOp):
    """Smooth each selected array in-place: boxcar, binomial, or Savitzky-Golay.

    Uses shared pure functions from :mod:`pyneuromatic.core.nm_math`.
    Edge effects at both ends of each array (half a window-width of points)
    are an inherent property of ``np.convolve`` with ``mode='same'``.

    Parameters:
        method: ``"boxcar"``, ``"binomial"``, or ``"savgol"``. Default ``"boxcar"``.
        window: Kernel width in points (odd int >= 3). Used by boxcar and savgol.
            Not used by binomial. Default 5.
        passes: Number of times to apply the kernel (int >= 1). Default 1.
            Used by boxcar and binomial; ignored by savgol.
        polyorder: Polynomial order for savgol (int >= 1, < window). Default 2.
    """

    name = "smooth"

    def __init__(
        self,
        method: str = "boxcar",
        window: int = 5,
        passes: int = 1,
        polyorder: int = 2,
    ) -> None:
        self.method = method
        self.window = window
        self.passes = passes
        self.polyorder = polyorder

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
    # Lifecycle

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Smooth *data* in-place."""
        import pyneuromatic.core.nm_math as nm_math
        y = data.nparray
        if y is None:
            return
        if self._method == "savgol" and self._passes > 1:
            nmh.history(
                "passes=%d ignored for savgol; savgol is applied once" % self._passes,
                title="ALERT",
                red=True,
            )
        if self._method == "boxcar":
            data.nparray = nm_math.smooth_boxcar(y, self._window, self._passes)
        elif self._method == "binomial":
            data.nparray = nm_math.smooth_binomial(y, self._passes)
        else:  # savgol
            data.nparray = nm_math.smooth_savgol(y, self._window, self._polyorder)
        self._add_op_note(data)

    def _op_params_str(self) -> str:
        effective_passes = 1 if self._method == "savgol" else self._passes
        return "method=%r, window=%r, passes=%r, polyorder=%r" % (
            self._method, self._window, effective_passes, self._polyorder
        )


# =========================================================================
# Filter
# =========================================================================


class NMMainOpFilter(NMMainOp):
    """Filter each array using Butterworth, Bessel, or notch filtering.

    All filters are applied zero-phase (forward-backward) via
    ``sosfiltfilt`` or ``filtfilt``, so no phase distortion is introduced.

    Three filter types:

    - ``"butterworth"``: maximally flat magnitude; general-purpose low-pass,
      high-pass, or band-pass.
    - ``"bessel"``: maximally flat group delay; preferred when waveform
      shape must be preserved (e.g. spike kinetics, EPSC rise times).
    - ``"notch"``: narrow band-stop at *cutoff* Hz; removes mains
      interference (50 or 60 Hz).

    Parameters:
        filter_type: ``"butterworth"`` (default), ``"bessel"``, or
            ``"notch"``.
        cutoff: Cutoff frequency in Hz.  For ``btype='bandpass'`` supply
            ``[low_hz, high_hz]``.  For ``"notch"`` this is the centre
            frequency to remove.
        order: Filter order (int >= 1).  Not used for ``"notch"``.
            Default 4.
        btype: ``"low"`` (default), ``"high"``, or ``"bandpass"``.
            Not used for ``"notch"``.
        q: Quality factor for ``"notch"`` — ratio of centre frequency to
            bandwidth; higher values give a narrower notch. Default 30.
        sample_rate: Sample rate in Hz.  If ``None`` (default), derived
            from ``xscale.delta`` and ``xscale.units`` at run time
            (assumes SI-prefixed time units, e.g. ``"ms"``).
    """

    name = "filter"

    def __init__(
        self,
        filter_type: str = "butterworth",
        cutoff: float | list[float] = 1000.0,
        order: int = 4,
        btype: str = "low",
        q: float = 30.0,
        sample_rate: float | None = None,
    ) -> None:
        self.filter_type = filter_type
        self.cutoff = cutoff
        self.order = order
        self.btype = btype
        self.q = q
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Properties

    @property
    def filter_type(self) -> str:
        """Filter type: ``'butterworth'``, ``'bessel'``, or ``'notch'``."""
        return self._filter_type

    @filter_type.setter
    def filter_type(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "filter_type", "string"))
        if value not in nm_math._VALID_FILTER_TYPES:
            raise ValueError(
                "filter_type must be one of %s, got %r"
                % (sorted(nm_math._VALID_FILTER_TYPES), value)
            )
        self._filter_type = value

    @property
    def cutoff(self) -> float | list[float]:
        """Cutoff frequency in Hz (float, or [low, high] for bandpass)."""
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: float | list[float]) -> None:
        if isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "cutoff", "float or list"))
        if isinstance(value, (int, float)):
            if value <= 0:
                raise ValueError("cutoff must be > 0, got %g" % value)
            self._cutoff = float(value)
        elif isinstance(value, list):
            self._cutoff = value
        else:
            raise TypeError(nmu.type_error_str(value, "cutoff", "float or list"))

    @property
    def order(self) -> int:
        """Filter order (int >= 1). Not used for notch."""
        return self._order

    @order.setter
    def order(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "order", "int"))
        if value < 1:
            raise ValueError("order must be >= 1, got %d" % value)
        self._order = value

    @property
    def btype(self) -> str:
        """Filter band type: ``'low'``, ``'high'``, or ``'bandpass'``. Not used for notch."""
        return self._btype

    @btype.setter
    def btype(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "btype", "string"))
        if value not in nm_math._VALID_FILTER_BTYPES:
            raise ValueError(
                "btype must be one of %s, got %r"
                % (sorted(nm_math._VALID_FILTER_BTYPES), value)
            )
        self._btype = value

    @property
    def q(self) -> float:
        """Quality factor for notch filter (float > 0). Not used for other types."""
        return self._q

    @q.setter
    def q(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "q", "float"))
        if value <= 0:
            raise ValueError("q must be > 0, got %g" % value)
        self._q = float(value)

    @property
    def sample_rate(self) -> float | None:
        """Sample rate in Hz. If None, derived from xscale at run time."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: float | None) -> None:
        if value is None:
            self._sample_rate = None
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "sample_rate", "float"))
        if value <= 0:
            raise ValueError("sample_rate must be > 0, got %g" % value)
        self._sample_rate = float(value)

    # ------------------------------------------------------------------
    # Core

    def _resolve_sample_rate(self, data: NMData) -> float:
        """Return sample rate in Hz from parameter or xscale."""
        if self._sample_rate is not None:
            return self._sample_rate
        delta = data.xscale.delta
        units = data.xscale.units
        if units:
            factor = nm_math.si_scale_factor(units, "s")
            return 1.0 / (delta * factor)
        return 1.0 / delta  # assume seconds

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Filter *data* in-place."""
        y = data.nparray
        if y is None:
            return
        sr = self._resolve_sample_rate(data)
        if self._filter_type == "butterworth":
            data.nparray = nm_math.filter_butterworth(
                y, self._cutoff, sr, self._order, self._btype
            )
        elif self._filter_type == "bessel":
            data.nparray = nm_math.filter_bessel(
                y, self._cutoff, sr, self._order, self._btype
            )
        else:  # notch
            data.nparray = nm_math.filter_notch(y, self._cutoff, sr, self._q)
        self._add_op_note(data, self._op_params_str())

    def _op_params_str(self) -> str:
        if self._filter_type == "notch":
            return "filter_type=%r, cutoff=%r, q=%r" % (
                self._filter_type, self._cutoff, self._q
            )
        return "filter_type=%r, cutoff=%r, order=%r, btype=%r" % (
            self._filter_type, self._cutoff, self._order, self._btype
        )


# =========================================================================
# Resample
# =========================================================================


class NMMainOpResample(NMMainOp):
    """Resample each array to a new sample interval using polyphase filtering.

    Wraps ``scipy.signal.resample_poly`` via :func:`nm_math.resample`,
    which applies an anti-aliasing FIR filter making it correct for both
    upsampling and downsampling.  Updates ``xscale.delta`` after resampling;
    ``xscale.start`` is unchanged.

    Parameters:
        delta: New sample interval in the same units as ``xscale.delta``.
            Must be > 0.
    """

    name = "resample"

    def __init__(self, delta: float) -> None:
        self.delta = delta

    # ------------------------------------------------------------------
    # Properties

    @property
    def delta(self) -> float:
        """New sample interval in the same units as ``xscale.delta``."""
        return self._delta

    @delta.setter
    def delta(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "delta", "float"))
        if value <= 0:
            raise ValueError("delta must be > 0, got %g" % value)
        self._delta = float(value)

    # ------------------------------------------------------------------
    # Core

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Resample *data* in-place."""
        y = data.nparray
        if y is None:
            return
        data.nparray = nm_math.resample(y, data.xscale.delta, self._delta)
        data.xscale.delta = self._delta
        self._add_op_note(data, self._op_params_str())

    def _op_params_str(self) -> str:
        return "delta=%r" % self._delta


# =========================================================================
# Interpolate
# =========================================================================

_VALID_INTERP_X_SOURCES: frozenset[str] = frozenset({"common", "template"})
_VALID_INTERP_X_EXTENT: frozenset[str] = frozenset({"overlap", "expand"})


class NMMainOpInterpolate(NMMainOp):
    """Re-grid each array onto a common x-axis via interpolation.

    Useful when arrays were recorded at slightly different sample rates and
    need to be interpolated onto a shared x-axis before averaging or comparison.
    Uses :func:`nm_math.interpolate` (``scipy.interpolate``).

    Two x-axis sources:

    - ``"common"``: derive the x-axis target from the data themselves.
      Requires ``run_all()`` (or ``NMToolMain``) so that all arrays are
      visible before interpolation begins.
    - ``"template"``: use the xscale of a named NMData array in the same
      folder as the template x-axis.

    When *x_source* is ``"common"``, *x_extent* controls the x-axis range:

    - ``"overlap"`` (default): start = max of all array starts, end = min
      of all array ends — only the region shared by every array is kept.
    - ``"expand"``: start = min of all array starts, end = max of all array
      ends — the full union is used.  Points outside an array's original
      range are filled with NaN.

    Parameters:
        method: ``"linear"`` (default) or ``"cubic"``.
        x_source: ``"common"`` (default) or ``"template"``.
        x_extent: ``"overlap"`` (default) or ``"expand"``.
            Only used when *x_source* is ``"common"``.
        template_name: Name of the NMData template array.  Required when
            *x_source* is ``"template"``.
    """

    name = "interpolate"

    def __init__(
        self,
        method: str = "linear",
        x_source: str = "common",
        x_extent: str = "overlap",
        template_name: str | None = None,
    ) -> None:
        self.method = method
        self.x_source = x_source
        self.x_extent = x_extent
        self.template_name = template_name
        self._x_new: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties

    @property
    def method(self) -> str:
        """Interpolation method: ``'linear'`` or ``'cubic'``."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "method", "string"))
        if value not in nm_math._VALID_INTERPOLATE_METHODS:
            raise ValueError(
                "method must be one of %s, got %r"
                % (sorted(nm_math._VALID_INTERPOLATE_METHODS), value)
            )
        self._method = value

    @property
    def x_source(self) -> str:
        """X-axis source: ``'common'`` or ``'template'``."""
        return self._x_source

    @x_source.setter
    def x_source(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "x_source", "string"))
        if value not in _VALID_INTERP_X_SOURCES:
            raise ValueError(
                "x_source must be one of %s, got %r"
                % (sorted(_VALID_INTERP_X_SOURCES), value)
            )
        self._x_source = value

    @property
    def x_extent(self) -> str:
        """X-axis extent when x_source='common': ``'overlap'`` or ``'expand'``."""
        return self._x_extent

    @x_extent.setter
    def x_extent(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "x_extent", "string"))
        if value not in _VALID_INTERP_X_EXTENT:
            raise ValueError(
                "x_extent must be one of %s, got %r"
                % (sorted(_VALID_INTERP_X_EXTENT), value)
            )
        self._x_extent = value

    @property
    def template_name(self) -> str | None:
        """Name of NMData template array (used when x_source='template')."""
        return self._template_name

    @template_name.setter
    def template_name(self, value: str | None) -> None:
        if value is not None and not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "template_name", "string"))
        self._template_name = value

    # ------------------------------------------------------------------
    # Core

    def run_init(self) -> None:
        self._x_new = None
        if self._x_source == "common":
            self._x_new = self._compute_common_x()
        elif self._x_source == "template":
            self._x_new = self._compute_template_x()

    def _compute_common_x(self) -> np.ndarray | None:
        """Derive common x-axis from all data items.

        ``x_extent='overlap'``: intersection of all x-ranges (no NaNs).
        ``x_extent='expand'``: union of all x-ranges (NaN where an array has
        no data).
        """
        items = getattr(self, "_data_items", None)
        if not items:
            return None
        starts, ends, deltas = [], [], []
        for data, _ in items:
            y = data.nparray
            if y is None or len(y) == 0:
                continue
            s = data.xscale.start
            d = data.xscale.delta
            starts.append(s)
            ends.append(s + (len(y) - 1) * d)
            deltas.append(d)
        if not starts:
            return None
        if self._x_extent == "expand":
            x_start = min(starts)   # union: earliest start
            x_end = max(ends)       # union: latest end
        else:
            x_start = max(starts)   # overlap: latest start
            x_end = min(ends)       # overlap: earliest end
        x_delta = min(deltas)       # finest resolution
        if x_end <= x_start:
            return None
        n = round((x_end - x_start) / x_delta) + 1
        return np.linspace(x_start, x_end, n)

    def _compute_template_x(self) -> np.ndarray | None:
        """Derive x-axis from the named template NMData array."""
        if not self._template_name:
            return None
        folder = getattr(self, "_folder", None)
        if folder is None:
            return None
        d = folder.data.get(self._template_name)
        if d is None or d.nparray is None:
            return None
        n = len(d.nparray)
        return np.linspace(
            d.xscale.start,
            d.xscale.start + (n - 1) * d.xscale.delta,
            n,
        )

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Interpolate *data* in-place onto the x-axis target."""
        y = data.nparray
        if y is None:
            return
        if self._x_new is None:
            raise ValueError(
                "NMMainOpInterpolate: x_new not set — call run_all() or "
                "run_init() before run(), or check x_source/template_name"
            )
        n = len(y)
        x_old = np.linspace(
            data.xscale.start,
            data.xscale.start + (n - 1) * data.xscale.delta,
            n,
        )
        data.nparray = nm_math.interpolate(y, x_old, self._x_new,
                                           method=self._method)
        data.xscale.start = float(self._x_new[0])
        data.xscale.delta = float(self._x_new[1] - self._x_new[0])
        self._add_op_note(data, self._op_params_str())

    def _op_params_str(self) -> str:
        return "method=%r, x_source=%r, x_extent=%r, template_name=%r" % (
            self._method, self._x_source, self._x_extent, self._template_name
        )


# =========================================================================
# Align
# =========================================================================

_VALID_ALIGN_TARGETS: frozenset[str] = frozenset({"mean", "min", "max"})


class NMMainOpAlign(NMMainOp):
    """Shift xscale.start of each array so reference xvalues align to a target.

    For each array i, computes ``shift = target_value - xvalues[i]`` and sets::

        data.xscale.start += shift

    The data values and ``xscale.delta`` are unchanged.  To re-grid onto a
    common x-axis after aligning, run :class:`NMMainOpInterpolate` afterwards.

    Parameters:
        xvalues: Reference xvalue(s), one per array:

            - ``float``: the same xvalue is applied to every array.
            - ``list[float]``: one per array in order; length must match
              the number of data items.
            - ``dict[str, float]``: lookup by data name; arrays whose name
              is not in the dict are silently skipped.

        target: Where to place the reference after alignment:

            - ``float`` (default ``0.0``): a fixed xvalue.
            - ``"mean"`` / ``"min"`` / ``"max"``: computed from *xvalues*
              at runtime.
    """

    name = "align"

    def __init__(
        self,
        xvalues: float | list[float] | dict[str, float] = 0.0,
        target: float | str = 0.0,
    ) -> None:
        self.xvalues = xvalues
        self.target = target
        self._index: int = 0
        self._target_xvalue: float = 0.0

    # ------------------------------------------------------------------
    # Properties

    @property
    def xvalues(self) -> float | list[float] | dict[str, float]:
        """Reference xvalues: float, list[float], or dict[str, float]."""
        return self._xvalues

    @xvalues.setter
    def xvalues(self, value: float | list[float] | dict[str, float]) -> None:
        if isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "xvalues", "float, list, or dict"))
        if isinstance(value, (int, float)):
            self._xvalues = float(value)
        elif isinstance(value, list):
            self._xvalues = value
        elif isinstance(value, dict):
            self._xvalues = value
        else:
            raise TypeError(nmu.type_error_str(value, "xvalues", "float, list, or dict"))

    @property
    def target(self) -> float | str:
        """Alignment target: float or ``'mean'`` / ``'min'`` / ``'max'``."""
        return self._target

    @target.setter
    def target(self, value: float | str) -> None:
        if isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "target", "float or string"))
        if isinstance(value, (int, float)):
            self._target = float(value)
        elif isinstance(value, str):
            if value not in _VALID_ALIGN_TARGETS:
                raise ValueError(
                    "target must be a float or one of %s, got %r"
                    % (sorted(_VALID_ALIGN_TARGETS), value)
                )
            self._target = value
        else:
            raise TypeError(nmu.type_error_str(value, "target", "float or string"))

    # ------------------------------------------------------------------
    # Core

    def run_init(self) -> None:
        self._index = 0
        if isinstance(self._xvalues, list) and len(self._xvalues) != self._n_items:
            raise IndexError(
                "NMMainOpAlign: xvalues list length must match number of "
                "data items (need %d, got %d)" % (self._n_items, len(self._xvalues))
            )
        if isinstance(self._target, (int, float)):
            self._target_xvalue = float(self._target)
        else:
            values = self._collect_xvalues()
            if self._target == "mean":
                self._target_xvalue = float(np.mean(values))
            elif self._target == "min":
                self._target_xvalue = float(np.min(values))
            else:  # "max"
                self._target_xvalue = float(np.max(values))

    def _collect_xvalues(self) -> list[float]:
        """Flatten xvalues to a list of floats for target computation."""
        if isinstance(self._xvalues, (int, float)):
            return [self._xvalues] * self._n_items
        if isinstance(self._xvalues, list):
            return [float(v) for v in self._xvalues]
        return [float(v) for v in self._xvalues.values()]  # dict

    def _get_xvalue(self, name: str) -> float | None:
        if isinstance(self._xvalues, (int, float)):
            return self._xvalues
        if isinstance(self._xvalues, list):
            v = float(self._xvalues[self._index])
            self._index += 1
            return v
        return self._xvalues.get(name)  # dict: None → skip

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Shift *data* xscale.start in-place."""
        xvalue = self._get_xvalue(data.name)
        if xvalue is None:
            return  # dict mode: name not found — skip
        xshift = self._target_xvalue - xvalue
        data.xscale.start = data.xscale.start + xshift
        self._add_op_note(data, "xvalue=%.6g, target=%.6g, xshift=%.6g" % (
            xvalue, self._target_xvalue, xshift))

    def _op_params_str(self) -> str:
        return "xvalues=%r, target=%r" % (self._xvalues, self._target)


# =========================================================================
# Normalize
# =========================================================================


class NMMainOpNormalize(NMMainOp):
    """Rescale each array so a low reference maps to norm_min and a high reference maps to norm_max.

    Two independent xscale windows are used:

    - Window 1 (``x0_min``/``x1_min``) computes the "low" reference via
      ``fxn1`` (``"mean"``, ``"min"``, or ``"mean@min"``).
    - Window 2 (``x0_max``/``x1_max``) computes the "high" reference via
      ``fxn2`` (``"mean"``, ``"max"``, or ``"mean@max"``).

    Two modes (matching NMMainOpBaseline):

    - **per_array**: Each array is normalized to its own references.
    - **average**: Per-channel mean references are computed across all arrays,
      then applied to every array in that channel.

    Parameters:
        x0_min: Window 1 start in xscale units (default 0.0).
        x1_min: Window 1 end in xscale units (default 0.0).
        fxn1: Function for the low reference: ``"mean"``, ``"min"``, or
            ``"mean@min"`` (default ``"mean"``).
        n_mean1: Points around min for ``mean@min`` (default 1).
        x0_max: Window 2 start in xscale units (default 0.0).
        x1_max: Window 2 end in xscale units (default 0.0).
        fxn2: Function for the high reference: ``"mean"``, ``"max"``, or
            ``"mean@max"`` (default ``"mean"``).
        n_mean2: Points around max for ``mean@max`` (default 1).
        norm_min: Target normalized minimum (default 0.0).
        norm_max: Target normalized maximum (default 1.0).
        mode: ``"per_array"`` (default) or ``"average"``.
    """

    name = "normalize"

    _VALID_FXN1 = {"mean", "min", "mean@min"}
    _VALID_FXN2 = {"mean", "max", "mean@max"}
    _VALID_MODES = {"per_array", "average"}

    def __init__(
        self,
        x0_min: float = 0.0,
        x1_min: float = 0.0,
        fxn1: str = "mean",
        n_mean1: int = 1,
        x0_max: float = 0.0,
        x1_max: float = 0.0,
        fxn2: str = "mean",
        n_mean2: int = 1,
        norm_min: float = 0.0,
        norm_max: float = 1.0,
        mode: str = "per_array",
    ) -> None:
        self.x0_min = x0_min
        self.x1_min = x1_min
        self.fxn1 = fxn1
        self.n_mean1 = n_mean1
        self.x0_max = x0_max
        self.x1_max = x1_max
        self.fxn2 = fxn2
        self.n_mean2 = n_mean2
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.mode = mode

    # ------------------------------------------------------------------
    # Properties

    @property
    def x0_min(self) -> float:
        """Window 1 start (xscale units)."""
        return self._x0_min

    @x0_min.setter
    def x0_min(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x0_min", "float"))
        if math.isnan(float(value)):
            raise ValueError("x0_min must not be NaN")
        self._x0_min = float(value)

    @property
    def x1_min(self) -> float:
        """Window 1 end (xscale units)."""
        return self._x1_min

    @x1_min.setter
    def x1_min(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x1_min", "float"))
        if math.isnan(float(value)):
            raise ValueError("x1_min must not be NaN")
        self._x1_min = float(value)

    @property
    def fxn1(self) -> str:
        """Low-reference function: ``'mean'``, ``'min'``, or ``'mean@min'``."""
        return self._fxn1

    @fxn1.setter
    def fxn1(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "fxn1", "string"))
        if value not in self._VALID_FXN1:
            raise ValueError(
                "fxn1 must be one of %s, got %r" % (sorted(self._VALID_FXN1), value)
            )
        self._fxn1 = value

    @property
    def n_mean1(self) -> int:
        """Points around min for ``mean@min`` (default 1)."""
        return self._n_mean1

    @n_mean1.setter
    def n_mean1(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "n_mean1", "int"))
        if value < 1:
            raise ValueError("n_mean1 must be >= 1, got %d" % value)
        self._n_mean1 = value

    @property
    def x0_max(self) -> float:
        """Window 2 start (xscale units)."""
        return self._x0_max

    @x0_max.setter
    def x0_max(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x0_max", "float"))
        if math.isnan(float(value)):
            raise ValueError("x0_max must not be NaN")
        self._x0_max = float(value)

    @property
    def x1_max(self) -> float:
        """Window 2 end (xscale units)."""
        return self._x1_max

    @x1_max.setter
    def x1_max(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x1_max", "float"))
        if math.isnan(float(value)):
            raise ValueError("x1_max must not be NaN")
        self._x1_max = float(value)

    @property
    def fxn2(self) -> str:
        """High-reference function: ``'mean'``, ``'max'``, or ``'mean@max'``."""
        return self._fxn2

    @fxn2.setter
    def fxn2(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "fxn2", "string"))
        if value not in self._VALID_FXN2:
            raise ValueError(
                "fxn2 must be one of %s, got %r" % (sorted(self._VALID_FXN2), value)
            )
        self._fxn2 = value

    @property
    def n_mean2(self) -> int:
        """Points around max for ``mean@max`` (default 1)."""
        return self._n_mean2

    @n_mean2.setter
    def n_mean2(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "n_mean2", "int"))
        if value < 1:
            raise ValueError("n_mean2 must be >= 1, got %d" % value)
        self._n_mean2 = value

    @property
    def norm_min(self) -> float:
        """Target normalized minimum."""
        return self._norm_min

    @norm_min.setter
    def norm_min(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "norm_min", "float"))
        self._norm_min = float(value)

    @property
    def norm_max(self) -> float:
        """Target normalized maximum."""
        return self._norm_max

    @norm_max.setter
    def norm_max(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "norm_max", "float"))
        self._norm_max = float(value)

    @property
    def mode(self) -> str:
        """Normalization mode: ``'per_array'`` or ``'average'``."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "mode", "string"))
        if value not in self._VALID_MODES:
            raise ValueError(
                "mode must be one of %s, got %r" % (sorted(self._VALID_MODES), value)
            )
        self._mode = value

    # ------------------------------------------------------------------
    # Validation helper

    def _validate_windows(self) -> None:
        if self._x1_min < self._x0_min:
            raise ValueError(
                "x1_min (%g) must be >= x0_min (%g)" % (self._x1_min, self._x0_min)
            )
        if self._x1_max < self._x0_max:
            raise ValueError(
                "x1_max (%g) must be >= x0_max (%g)" % (self._x1_max, self._x0_max)
            )

    # ------------------------------------------------------------------
    # Internal helpers

    def _apply(self, arr: np.ndarray, ref_min: float, ref_max: float) -> np.ndarray:
        """Apply normalization formula to arr."""
        range = ref_max - ref_min
        if range == 0:
            return np.full_like(arr, self._norm_min)
        return (arr - ref_min) / range * (self._norm_max - self._norm_min) + self._norm_min

    def _note_str(self, ref_min: float, ref_max: float, channel_name: str | None = None) -> str:
        note = "%s(%s" % (self._op_name, self._op_params_str())
        if channel_name is not None:
            note += ", channel=%s" % channel_name
        note += ", ref_min=%.6g, ref_max=%.6g)" % (ref_min, ref_max)
        return note

    # ------------------------------------------------------------------
    # Lifecycle

    def run_init(self) -> None:
        """Reset per-run accumulators."""
        self._validate_windows()
        self._ref_min_accum: dict[str, list[float]] = {}
        self._ref_max_accum: dict[str, list[float]] = {}
        self._data_refs: dict[str, list[NMData]] = {}

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Compute references and (optionally) normalize one array.

        Args:
            data: The NMData object to process.
            channel_name: Channel name from the selection context, or None
                (parsed from data.name as a fallback).
        """
        if not isinstance(data.nparray, np.ndarray):
            return

        if channel_name is None:
            parsed = nmu.parse_data_name(data.name)
            channel_name = parsed[1] if parsed is not None else "A"

        arr = data.nparray.astype(float)
        xd = data.xscale.to_dict()
        sl1 = nm_math.xscale_window_to_slice(arr, xd, self._x0_min, self._x1_min)
        sl2 = nm_math.xscale_window_to_slice(arr, xd, self._x0_max, self._x1_max)
        ref_min = nm_math.compute_ref_value(arr[sl1], self._fxn1, self._n_mean1)
        ref_max = nm_math.compute_ref_value(arr[sl2], self._fxn2, self._n_mean2)

        if self._mode == "per_array":
            data.nparray = self._apply(arr, ref_min, ref_max)
            self._add_note(data, self._note_str(ref_min, ref_max))
        else:  # "average"
            self._ref_min_accum.setdefault(channel_name, []).append(ref_min)
            self._ref_max_accum.setdefault(channel_name, []).append(ref_max)
            self._data_refs.setdefault(channel_name, []).append(data)

    def run_finish(
        self,
        folder: NMFolder | None = None,
        prefix: str | None = None,
    ) -> None:
        """Apply averaged references (average mode only).

        In ``per_array`` mode this is a no-op (normalization was done in
        ``run()``).  In ``average`` mode the mean of all per-array references
        for each channel is computed and applied to every array in that channel.
        """
        if self._mode == "per_array":
            return
        for channel_name, ref_mins in self._ref_min_accum.items():
            avg_ref_min = float(np.nanmean(ref_mins))
            avg_ref_max = float(np.nanmean(self._ref_max_accum[channel_name]))
            for d in self._data_refs[channel_name]:
                arr = d.nparray.astype(float)
                d.nparray = self._apply(arr, avg_ref_min, avg_ref_max)
                self._add_note(d, self._note_str(avg_ref_min, avg_ref_max, channel_name))

    def _op_params_str(self) -> str:
        return (
            "x0_min=%r, x1_min=%r, fxn1=%r, n_mean1=%d, "
            "x0_max=%r, x1_max=%r, fxn2=%r, n_mean2=%d, "
            "norm_min=%r, norm_max=%r, mode=%r"
        ) % (
            self._x0_min, self._x1_min, self._fxn1, self._n_mean1,
            self._x0_max, self._x1_max, self._fxn2, self._n_mean2,
            self._norm_min, self._norm_max, self._mode,
        )


# =========================================================================
# DFOF
# =========================================================================


class NMMainOpDFOF(NMMainOp):
    """Compute dF/F₀ = (F − F₀) / F₀ in-place for each data array.

    F₀ is the mean fluorescence over the baseline xscale window [x0, x1].
    The array name is unchanged; a note records the transformation and F₀.
    After transformation, ``yscale.label`` is set to ``"dF/F"`` and
    ``yscale.units`` to ``""`` (dimensionless).

    Two modes are supported:

    - **per_array**: Each array's own F₀ (mean of the window) is used
      independently.
    - **average**: A single shared F₀ per channel is computed as the mean
      of all per-array F₀ values for that channel, then applied to every
      array in that channel.

    Parameters:
        x0:          Baseline window start in x-axis units.  Default ``-inf``
                     (start of array).
        x1:          Baseline window end in x-axis units.  Default ``+inf``
                     (end of array).
        mode:        ``"per_array"`` (default) or ``"average"``.
        ignore_nans: If True (default) use ``np.nanmean``; otherwise
                     ``np.mean`` (NaN propagates to the result).
    """

    name = "dfof"

    _VALID_MODES = {"per_array", "average"}

    def __init__(
        self,
        x0: float = -math.inf,
        x1: float = math.inf,
        mode: str = "per_array",
        ignore_nans: bool = True,
    ) -> None:
        self.x0 = x0
        self.x1 = x1
        self.mode = mode
        self.ignore_nans = ignore_nans

    # ------------------------------------------------------------------
    # Properties

    @property
    def x0(self) -> float:
        """Baseline window start (x-axis units)."""
        return self._x0

    @x0.setter
    def x0(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x0", "float"))
        if math.isnan(float(value)):
            raise ValueError("x0 must not be NaN")
        self._x0 = float(value)

    @property
    def x1(self) -> float:
        """Baseline window end (x-axis units)."""
        return self._x1

    @x1.setter
    def x1(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x1", "float"))
        if math.isnan(float(value)):
            raise ValueError("x1 must not be NaN")
        self._x1 = float(value)

    @property
    def mode(self) -> str:
        """F₀ mode: ``'per_array'`` or ``'average'``."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "mode", "string"))
        if value not in self._VALID_MODES:
            raise ValueError(
                "mode must be one of %s, got %r" % (sorted(self._VALID_MODES), value)
            )
        self._mode = value

    @property
    def ignore_nans(self) -> bool:
        """If True, NaN values are excluded from F₀ mean (np.nanmean)."""
        return self._ignore_nans

    @ignore_nans.setter
    def ignore_nans(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "ignore_nans", "boolean"))
        self._ignore_nans = value

    # ------------------------------------------------------------------
    # Validation helper

    def _validate_window(self) -> None:
        if self._x1 < self._x0:
            raise ValueError(
                "x1 (%g) must be >= x0 (%g)" % (self._x1, self._x0)
            )

    # ------------------------------------------------------------------
    # Lifecycle

    def run_init(self) -> None:
        """Reset per-run accumulators."""
        self._validate_window()
        self._f0_accum: dict[str, list[float]] = {}
        self._data_refs: dict[str, list[NMData]] = {}

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Compute and (optionally) apply dF/F₀ for one array.

        Args:
            data: The NMData object to process.
            channel_name: Channel name from the selection context, or None
                (parsed from data.name as a fallback).
        """
        if not isinstance(data.nparray, np.ndarray):
            return

        if channel_name is None:
            parsed = nmu.parse_data_name(data.name)
            channel_name = parsed[1] if parsed is not None else "A"

        arr = data.nparray.astype(float)
        sl = nm_math.xscale_window_to_slice(
            arr, data.xscale.to_dict(), self._x0, self._x1
        )
        segment = arr[sl]
        if len(segment) == 0:
            f0 = 0.0
        elif self._ignore_nans:
            f0 = float(np.nanmean(segment))
        else:
            f0 = float(np.mean(segment))

        if self._mode == "per_array":
            self._apply(data, arr, f0, "per_array")
        else:  # "average"
            self._f0_accum.setdefault(channel_name, []).append(f0)
            self._data_refs.setdefault(channel_name, []).append(data)

    def run_finish(
        self,
        folder: NMFolder | None = None,
        prefix: str | None = None,
    ) -> None:
        """Apply averaged F₀ (average mode only).

        In ``per_array`` mode this is a no-op (transform was done in ``run()``).
        In ``average`` mode the mean of all per-array F₀ values for each channel
        is computed and used to transform every array in that channel.
        """
        if self._mode == "per_array":
            return
        for channel_name, f0_list in self._f0_accum.items():
            mean_f0 = float(
                np.nanmean(f0_list) if self._ignore_nans else np.mean(f0_list)
            )
            for d in self._data_refs[channel_name]:
                self._apply(d, d.nparray.astype(float), mean_f0, "average",
                            channel_name=channel_name)

    def _apply(
        self,
        data: NMData,
        arr: np.ndarray,
        f0: float,
        mode_label: str,
        channel_name: str | None = None,
    ) -> None:
        data.nparray = nm_math.apply_dfof(arr, f0)
        data.yscale.label = "dF/F"
        data.yscale.units = ""
        if channel_name is not None:
            self._add_op_note(data, "%s, channel=%s, F0=%.6g" % (
                self._op_params_str(), channel_name, f0))
        else:
            self._add_op_note(data, "%s, F0=%.6g" % (self._op_params_str(), f0))

    def _op_params_str(self) -> str:
        return "x0=%r, x1=%r, mode=%r, ignore_nans=%r" % (
            self._x0, self._x1, self._mode, self._ignore_nans)


# =========================================================================
# Rescale
# =========================================================================


class NMMainOpRescale(NMMainOp):
    """Rescale each data array between SI-prefixed unit variants in-place.

    Multiplies the array by the power-of-10 factor implied by the unit
    conversion and updates ``yscale.units``.

    Supported base units include ``"V"`` (volts), ``"A"`` (amperes),
    ``"Ohm"`` or ``"Ω"`` (ohms), and ``"s"`` (seconds).

    Parameters:
        to_units:   Target units string (e.g. ``"nA"``).  Required; must
                    not be empty at runtime.
        from_units: Source units string.  Defaults to ``None``, which
                    means the source units are read from
                    ``data.yscale.units`` at runtime.  Raises
                    ``ValueError`` if that field is also empty.
    """

    name = "rescale"

    def __init__(
        self,
        to_units: str = "",
        from_units: str | None = None,
    ) -> None:
        self.to_units = to_units
        self.from_units = from_units

    # ------------------------------------------------------------------
    # Properties

    @property
    def to_units(self) -> str:
        return self._to_units

    @to_units.setter
    def to_units(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "to_units", "string"))
        self._to_units = value

    @property
    def from_units(self) -> str | None:
        return self._from_units

    @from_units.setter
    def from_units(self, value: str | None) -> None:
        if value is not None and not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "from_units", "string or None"))
        self._from_units = value

    # ------------------------------------------------------------------
    # Validation

    def _validate(self) -> None:
        if not self._to_units:
            raise ValueError("to_units must not be empty")

    # ------------------------------------------------------------------
    # Lifecycle

    def run_init(self) -> None:
        self._validate()

    def run(self, data: NMData, channel_name: str | None = None) -> None:
        if not isinstance(data.nparray, np.ndarray):
            return

        from_units = self._from_units if self._from_units is not None \
            else data.yscale.units
        if not from_units:
            raise ValueError(
                "from_units not set and data.yscale.units is empty for %r"
                % data.name
            )

        factor = nm_math.si_scale_factor(from_units, self._to_units)
        data.nparray = data.nparray.astype(float) * factor
        data.yscale.units = self._to_units
        self._add_op_note(data, "%s, from=%s, factor=%.6g" % (
            self._op_params_str(), from_units, factor))

    def _op_params_str(self) -> str:
        return "to_units=%r, from_units=%r" % (self._to_units, self._from_units)


# =========================================================================
# RescaleX
# =========================================================================


class NMMainOpRescaleX(NMMainOp):
    """Rescale the x-axis between SI-prefixed unit variants in-place.

    Multiplies ``xscale.start`` and ``xscale.delta`` by the power-of-10
    factor implied by the unit conversion and updates ``xscale.units``.

    Supported base units include ``"s"`` (seconds), ``"m"`` (metres),
    ``"V"`` (volts), ``"A"`` (amperes), and ``"Hz"`` (hertz).

    Parameters:
        to_units:   Target units string (e.g. ``"s"``).  Required; must
                    not be empty at runtime.
        from_units: Source units string.  Defaults to ``None``, which
                    means the source units are read from
                    ``data.xscale.units`` at runtime.  Raises
                    ``ValueError`` if that field is also empty.
    """

    name = "rescale_x"

    def __init__(
        self,
        to_units: str = "",
        from_units: str | None = None,
    ) -> None:
        self.to_units = to_units
        self.from_units = from_units

    # ------------------------------------------------------------------
    # Properties

    @property
    def to_units(self) -> str:
        return self._to_units

    @to_units.setter
    def to_units(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "to_units", "string"))
        self._to_units = value

    @property
    def from_units(self) -> str | None:
        return self._from_units

    @from_units.setter
    def from_units(self, value: str | None) -> None:
        if value is not None and not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "from_units", "string or None"))
        self._from_units = value

    # ------------------------------------------------------------------
    # Validation

    def _validate(self) -> None:
        if not self._to_units:
            raise ValueError("to_units must not be empty")

    # ------------------------------------------------------------------
    # Lifecycle

    def run_init(self) -> None:
        self._validate()

    def run(self, data: NMData, channel_name: str | None = None) -> None:
        from_units = self._from_units if self._from_units is not None \
            else data.xscale.units
        if not from_units:
            raise ValueError(
                "from_units not set and data.xscale.units is empty for %r"
                % data.name
            )

        factor = nm_math.si_scale_factor(from_units, self._to_units)
        data.xscale.start = data.xscale.start * factor
        data.xscale.delta = data.xscale.delta * factor
        data.xscale.units = self._to_units
        self._add_op_note(data, "%s, from=%s, factor=%.6g" % (
            self._op_params_str(), from_units, factor))

    def _op_params_str(self) -> str:
        return "to_units=%r, from_units=%r" % (self._to_units, self._from_units)


# =========================================================================
# Accumulate (base for Average, Sum, SumSqr, Min, Max)
# =========================================================================


class NMMainOpAccumulate(NMMainOp):
    """Base class for ops that accumulate arrays and reduce them per channel.

    Subclasses set two class attributes and override ``_reduce()``:

    - ``_output_prefix``: prefix for the output array name (e.g. ``"Avg_"``).
    - ``_reduce(stack)``: compute the per-channel result from the stacked array.

    The full lifecycle is handled here: ``run_init`` resets buffers,
    ``run`` accumulates per-channel arrays, and ``run_finish`` reduces
    and writes one output array per channel.

    Parameters:
        ignore_nans: If True (default) use NaN-aware reductions
            (``np.nanmean``, ``np.nansum``, etc.); otherwise NaN propagates.
    """

    _output_prefix: str = ""

    def __init__(self, ignore_nans: bool = True) -> None:
        if not isinstance(ignore_nans, bool):
            raise TypeError(
                nmu.type_error_str(ignore_nans, "ignore_nans", "boolean")
            )
        self._ignore_nans = ignore_nans
        self._results: dict[str, str] = {}  # channel → output name
        self._out_prefix: str = self._output_prefix  # settable per-instance

    @property
    def ignore_nans(self) -> bool:
        """If True, NaN values are excluded from the reduction."""
        return self._ignore_nans

    @ignore_nans.setter
    def ignore_nans(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "ignore_nans", "boolean"))
        self._ignore_nans = value

    @property
    def out_prefix(self) -> str:
        """Prefix for output array names (default is op-specific, e.g. ``"Avg_"``)."""
        return self._out_prefix

    @out_prefix.setter
    def out_prefix(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "out_prefix", "string"))
        self._out_prefix = value

    @property
    def results(self) -> dict[str, str]:
        """Read-only dict mapping channel name → output NMData name."""
        return dict(self._results)

    def _op_params_str(self) -> str:
        """Default params for accumulate ops; includes ignore_nans."""
        return "ignore_nans=%r" % self._ignore_nans

    def _reduce(self, stack: np.ndarray) -> np.ndarray:
        """Reduce the stacked array to a single output array.

        Args:
            stack: 2-D array of shape (n_arrays, n_points).

        Returns:
            1-D reduced array of shape (n_points,).

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError(
            "%s._reduce() not implemented" % self.__class__.__name__
        )

    def run_init(self) -> None:
        """Reset accumulation state for a new run."""
        self._results.clear()
        self._accum: dict[str, list[np.ndarray]] = {}
        self._data_names: dict[str, list[str]] = {}
        self._xscales: dict[str, dict] = {}
        self._yscales: dict[str, dict] = {}
        self._parsed_prefix: str | None = None  # fallback if prefix not passed

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Accumulate one array into the per-channel buffer.

        Args:
            data: The NMData object to accumulate.
            channel_name: Channel name from the selection context, or None
                (parsed from data.name as a fallback).
        """
        if not isinstance(data.nparray, np.ndarray):
            return

        if channel_name is None:
            parsed = nmu.parse_data_name(data.name)
            channel_name = parsed[1] if parsed is not None else "A"

        if self._parsed_prefix is None:
            parsed = nmu.parse_data_name(data.name)
            self._parsed_prefix = parsed[0] if parsed is not None else ""

        if channel_name not in self._accum:
            self._accum[channel_name] = []
            self._xscales[channel_name] = data.xscale.to_dict()
            self._yscales[channel_name] = data.yscale.to_dict()

        self._accum[channel_name].append(data.nparray.astype(float).copy())
        self._data_names.setdefault(channel_name, []).append(data.name)

    def _epoch_str(self, cname: str) -> str:
        """Format the epoch list for the note string."""
        return nmu.format_epoch_string(self._data_names.get(cname, []))

    def _make_note_str(
        self,
        op_name: str,
        folder_name: str,
        ds_name: str,
        cname: str,
        epoch_str: str,
        n: int,
    ) -> str:
        """Build the note string for an output array."""
        return (
            "%s(folder=%s,dataseries=%s,channel=%s,epochs=%s,n_epochs=%d)"
            % (op_name, folder_name, ds_name, cname, epoch_str, n)
        )

    def _write_output_array(
        self,
        folder: NMFolder,
        out_name: str,
        arr: np.ndarray,
        op_name: str,
        folder_name: str,
        ds_name: str,
        cname: str,
        epoch_str: str,
        n: int,
    ) -> None:
        """Create or overwrite an output array in folder and add a note to it."""
        out_data = self._write_out_array(
            folder, out_name, arr,
            xscale=self._xscales[cname],
            yscale=self._yscales[cname],
        )
        if out_data is not None:
            self._add_note(
                out_data,
                self._make_note_str(op_name, folder_name, ds_name, cname, epoch_str, n),
            )

    def _process_channel(
        self,
        folder: NMFolder,
        pfx: str,
        cname: str,
        arrays: list[np.ndarray],
    ) -> None:
        """Reduce and write one output array for a single channel.

        Subclasses may override to write additional output arrays (e.g.
        Stdv, Var, SEM alongside the mean).

        Args:
            folder: Destination NMFolder.
            pfx: Dataseries prefix for the output array name.
            cname: Channel name.
            arrays: List of accumulated arrays for this channel.
        """
        min_len = min(len(a) for a in arrays)
        stack = np.stack([a[:min_len] for a in arrays])
        n = len(arrays)
        epoch_str = self._epoch_str(cname)
        arr = self._reduce(stack)
        base_name = self._out_prefix + pfx + cname
        out_name = self._make_out_name(folder, base_name)
        self._write_output_array(
            folder, out_name, arr, self._op_name, folder.name, pfx, cname, epoch_str, n
        )
        self._results[cname] = out_name

    def run_finish(
        self,
        folder: NMFolder | None = None,
        prefix: str | None = None,
    ) -> None:
        """Reduce and write one output array per channel to folder.

        Args:
            folder: Destination NMFolder for the output arrays.
            prefix: Dataseries name used as the output array prefix.  Falls
                back to the prefix parsed from the first array name if None.
        """
        if not self._accum or folder is None:
            return

        pfx = prefix if prefix is not None else (self._parsed_prefix or "")
        for cname, arrays in self._accum.items():
            self._process_channel(folder, pfx, cname, arrays)


# =========================================================================
# Average
# =========================================================================


class NMMainOpAverage(NMMainOpAccumulate):
    """Average selected data arrays per channel.

    Accumulates arrays by channel, truncates all arrays to the shortest
    length, and writes the mean as ``Avg_{prefix}{channel}``
    (e.g. ``Avg_RecordA``) into the source folder.

    Optionally also writes Stdv, Var, and/or SEM arrays (sample statistics,
    ``ddof=1``) when the corresponding ``compute_*`` parameter is True.

    Parameters:
        ignore_nans: If True (default) use ``np.nanmean``; otherwise
            ``np.mean`` (NaN propagates to the result).
        compute_stdv: If True, write ``Stdv_{prefix}{channel}`` (default False).
        compute_var: If True, write ``Var_{prefix}{channel}`` (default False).
        compute_sem: If True, write ``SEM_{prefix}{channel}`` (default False).
    """

    name = "average"
    _output_prefix = "Avg_"

    def __init__(
        self,
        ignore_nans: bool = True,
        compute_stdv: bool = False,
        compute_var: bool = False,
        compute_sem: bool = False,
    ) -> None:
        super().__init__(ignore_nans)
        self.compute_stdv = compute_stdv  # setters validate bool
        self.compute_var = compute_var
        self.compute_sem = compute_sem

    @property
    def compute_stdv(self) -> bool:
        """If True, write a Stdv output array alongside the mean."""
        return self._compute_stdv

    @compute_stdv.setter
    def compute_stdv(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "compute_stdv", "boolean"))
        self._compute_stdv = value

    @property
    def compute_var(self) -> bool:
        """If True, write a Var output array alongside the mean."""
        return self._compute_var

    @compute_var.setter
    def compute_var(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "compute_var", "boolean"))
        self._compute_var = value

    @property
    def compute_sem(self) -> bool:
        """If True, write a SEM output array alongside the mean."""
        return self._compute_sem

    @compute_sem.setter
    def compute_sem(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "compute_sem", "boolean"))
        self._compute_sem = value

    def _reduce(self, stack: np.ndarray) -> np.ndarray:
        return np.nanmean(stack, axis=0) if self._ignore_nans else np.mean(stack, axis=0)

    def _process_channel(
        self,
        folder: NMFolder,
        pfx: str,
        cname: str,
        arrays: list[np.ndarray],
    ) -> None:
        """Write mean array, then optionally Stdv/Var/SEM arrays."""
        super()._process_channel(folder, pfx, cname, arrays)
        if not (self._compute_stdv or self._compute_var or self._compute_sem):
            return
        # Derive sequence suffix from the main output name so companion
        # arrays always share the same suffix (e.g. _0, _1, ...).
        main_out_name = self._results[cname]
        base_main = self._out_prefix + pfx + cname
        suffix = main_out_name[len(base_main):]  # "" or "_0", "_1", ...
        min_len = min(len(a) for a in arrays)
        stack = np.stack([a[:min_len] for a in arrays])
        n = len(arrays)
        epoch_str = self._epoch_str(cname)
        std_fn = np.nanstd if self._ignore_nans else np.std
        std = std_fn(stack, axis=0, ddof=1)
        if self._compute_stdv:
            self._write_output_array(
                folder, "Stdv_" + pfx + cname + suffix, std,
                "NMStdv", folder.name, pfx, cname, epoch_str, n,
            )
        if self._compute_var:
            self._write_output_array(
                folder, "Var_" + pfx + cname + suffix, std ** 2,
                "NMVar", folder.name, pfx, cname, epoch_str, n,
            )
        if self._compute_sem:
            self._write_output_array(
                folder, "SEM_" + pfx + cname + suffix, std / np.sqrt(n),
                "NMSEM", folder.name, pfx, cname, epoch_str, n,
            )

    def _op_params_str(self) -> str:
        return (
            "ignore_nans=%r, compute_stdv=%r, compute_var=%r, compute_sem=%r"
        ) % (self._ignore_nans, self._compute_stdv, self._compute_var, self._compute_sem)


# =========================================================================
# Sum, SumSqr, Min, Max  (NMMainOpAccumulate subclasses)
# =========================================================================


class NMMainOpSum(NMMainOpAccumulate):
    """Sum selected data arrays point-by-point per channel.

    Writes ``Sum_{prefix}{channel}`` (e.g. ``Sum_RecordA``) to the folder.

    Parameters:
        ignore_nans: If True (default) use ``np.nansum`` (NaN treated as 0);
            otherwise ``np.sum`` (NaN propagates).
    """

    name = "sum"
    _output_prefix = "Sum_"

    def _reduce(self, stack: np.ndarray) -> np.ndarray:
        return np.nansum(stack, axis=0) if self._ignore_nans else np.sum(stack, axis=0)


class NMMainOpSumSqr(NMMainOpAccumulate):
    """Sum of squares of selected data arrays point-by-point per channel.

    Squares each array then sums, writing ``SumSqr_{prefix}{channel}``
    (e.g. ``SumSqr_RecordA``) to the folder.

    Parameters:
        ignore_nans: If True (default) use ``np.nansum`` on the squared array;
            otherwise ``np.sum`` (NaN propagates).
    """

    name = "sum_sqr"
    _output_prefix = "SumSqr_"

    def _reduce(self, stack: np.ndarray) -> np.ndarray:
        sq = stack ** 2
        return np.nansum(sq, axis=0) if self._ignore_nans else np.sum(sq, axis=0)


class NMMainOpMin(NMMainOpAccumulate):
    """Point-by-point minimum across selected data arrays per channel.

    Writes ``Min_{prefix}{channel}`` (e.g. ``Min_RecordA``) to the folder.

    Parameters:
        ignore_nans: If True (default) use ``np.nanmin`` (NaN ignored);
            otherwise ``np.min`` (NaN propagates).
    """

    name = "min"
    _output_prefix = "Min_"

    def _reduce(self, stack: np.ndarray) -> np.ndarray:
        return np.nanmin(stack, axis=0) if self._ignore_nans else np.min(stack, axis=0)


class NMMainOpMax(NMMainOpAccumulate):
    """Point-by-point maximum across selected data arrays per channel.

    Writes ``Max_{prefix}{channel}`` (e.g. ``Max_RecordA``) to the folder.

    Parameters:
        ignore_nans: If True (default) use ``np.nanmax`` (NaN ignored);
            otherwise ``np.max`` (NaN propagates).
    """

    name = "max"
    _output_prefix = "Max_"

    def _reduce(self, stack: np.ndarray) -> np.ndarray:
        return np.nanmax(stack, axis=0) if self._ignore_nans else np.max(stack, axis=0)


# =========================================================================
# Concatenate
# =========================================================================


class NMMainOpConcatenate(NMMainOpAccumulate):
    """Concatenate selected data arrays per channel into a single output array.

    In ``"1d"`` mode arrays are joined end-to-end (``np.concatenate``);
    unequal lengths are allowed.  In ``"2d"`` mode arrays are stacked as
    rows (``np.stack``); shorter arrays are padded with NaN to ``max_len``
    so no data is lost from longer arrays.

    Output array is written to the destination folder as
    ``Cat_{prefix}{channel}`` (e.g. ``Cat_RecordA``).

    Parameters:
        mode:        ``"1d"`` (default) or ``"2d"``.
        ignore_nans: Passed to the accumulate base (unused by concatenate
                     itself, kept for API consistency).
    """

    name = "concatenate"
    _output_prefix = "Cat_"
    _VALID_MODES: frozenset[str] = frozenset({"1d", "2d"})

    def __init__(self, mode: str = "1d", ignore_nans: bool = True) -> None:
        super().__init__(ignore_nans)
        self.mode = mode

    # ------------------------------------------------------------------
    # Properties

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "mode", "string"))
        if value not in self._VALID_MODES:
            raise ValueError(
                "mode must be one of %s, got %r" % (sorted(self._VALID_MODES), value)
            )
        self._mode = value

    # ------------------------------------------------------------------
    # Subclass contract

    def _reduce(self, stack: np.ndarray) -> np.ndarray:
        # Not called — _process_channel is fully overridden.
        return stack  # pragma: no cover

    def _process_channel(
        self,
        folder: NMFolder,
        pfx: str,
        cname: str,
        arrays: list[np.ndarray],
    ) -> None:
        if self._mode == "1d":
            arr = np.concatenate(arrays)
        else:  # "2d"
            max_len = max(len(a) for a in arrays)
            padded = []
            for a in arrays:
                deficit = max_len - len(a)
                if deficit:
                    a = np.concatenate([a, np.full(deficit, np.nan)])
                padded.append(a)
            arr = np.stack(padded)

        n = len(arrays)
        epoch_str = self._epoch_str(cname)
        base_name = self._out_prefix + pfx + cname
        out_name = self._make_out_name(folder, base_name)
        op_name = "%s_%s" % (self._op_name, self._mode)
        self._write_output_array(
            folder, out_name, arr, op_name,
            folder.name, pfx, cname, epoch_str, n,
        )
        self._results[cname] = out_name

    def _op_params_str(self) -> str:
        return "mode=%r, ignore_nans=%r" % (self._mode, self._ignore_nans)


# =========================================================================
# NMMainOpInequality
# =========================================================================


class NMMainOpInequality(NMMainOp):
    """Apply an inequality test point-by-point to each data array.

    Output is written as a new array ``IQ_{data.name}`` in the source
    folder (non-destructive).  Original data is unchanged.

    Args:
        op:            Operator string (see
                       :data:`~pyneuromatic.core.nm_math.VALID_INEQUALITY_OPS`).
        a:             Threshold / lower bound.
        b:             Upper bound; required for range ops, else None.
        binary_output: If True (default), output values are 1.0 (pass) or
                       0.0 (fail).  If False, output is the original value
                       where the condition is met and NaN elsewhere.
    """

    name = "inequality"

    def __init__(
        self,
        op: str = ">",
        a: float = 0.0,
        b: float | None = None,
        binary_output: bool = True,
    ) -> None:
        self.op = op
        self.a = a
        self.b = b
        self.binary_output = binary_output
        self._out_prefix: str = "IQ_"
        self._results: dict = {}

    @property
    def out_prefix(self) -> str:
        """Prefix for output array names (default ``"IQ_"``)."""
        return self._out_prefix

    @out_prefix.setter
    def out_prefix(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "out_prefix", "string"))
        self._out_prefix = value

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def op(self) -> str:
        return self._op

    @op.setter
    def op(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "op", "string"))
        if value not in nm_math.VALID_INEQUALITY_OPS:
            raise ValueError(
                "unknown operator %r. Valid ops: %s"
                % (value, sorted(nm_math.VALID_INEQUALITY_OPS))
            )
        self._op = value

    @property
    def a(self) -> float:
        return self._a

    @a.setter
    def a(self, value: float) -> None:
        if isinstance(value, bool):
            raise TypeError("a must be a number, not bool")
        if not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "a", "number"))
        self._a = float(value)

    @property
    def b(self) -> float | None:
        return self._b

    @b.setter
    def b(self, value: float | None) -> None:
        if value is None:
            self._b = None
            return
        if isinstance(value, bool):
            raise TypeError("b must be a number or None, not bool")
        if not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "b", "number"))
        self._b = float(value)

    @property
    def binary_output(self) -> bool:
        return self._binary_output

    @binary_output.setter
    def binary_output(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "binary_output", "bool"))
        self._binary_output = value

    @property
    def results(self) -> dict:
        """Per-output-array dict; populated after :meth:`run_all`."""
        return self._results

    # ------------------------------------------------------------------
    # NMMainOp interface
    # ------------------------------------------------------------------

    def run_init(self) -> None:
        self._results = {}
        if self._op in nm_math._RANGE_INEQUALITY_OPS and self._b is None:
            raise ValueError("range op %r requires b" % self._op)

    def run(self, data: NMData, channel_name: str | None = None) -> None:
        if not isinstance(data.nparray, np.ndarray):
            return
        arr = data.nparray.astype(float)
        mask = nm_math.inequality_mask(arr, self._op, self._a, self._b)
        result = (
            mask.astype(float)
            if self._binary_output
            else np.where(mask, arr, np.nan)
        )
        successes = int(np.sum(mask))
        condition = nm_math.inequality_condition_str(self._op, self._a, self._b)
        base_name = self._out_prefix + data.name
        out_name = self._make_out_name(self._folder, base_name) if self._folder is not None else base_name
        if self._folder is not None:
            xscale = {
                "start": data.xscale.start,
                "delta": data.xscale.delta,
                "label": data.xscale.label,
                "units": data.xscale.units,
            }
            yscale = {"label": data.yscale.label, "units": data.yscale.units}
            out_data = self._write_out_array(self._folder, out_name, result,
                                            xscale=xscale, yscale=yscale)
            if out_data is not None:
                self._add_op_note(out_data, condition)
        self._results[out_name] = {
            "successes": successes,
            "failures": len(arr) - successes,
            "condition": condition,
        }

    def _op_params_str(self) -> str:
        return "op=%r, a=%r, b=%r, binary_output=%r" % (
            self._op, self._a, self._b, self._binary_output)


# =========================================================================
# NMMainOpHistogram
# =========================================================================


class NMMainOpHistogram(NMMainOp):
    """Compute an amplitude histogram for each data array.

    For each array, all sample values are passed to ``numpy.histogram``
    and the resulting bin-counts array is written as a new array
    ``H_{data.name}`` in the source folder (non-destructive).

    The output array's xscale represents the histogram bins:
    ``xscale.start`` = left edge of the first bin,
    ``xscale.delta`` = uniform bin width.
    ``xscale.label`` / ``xscale.units`` are taken from the input
    array's y-scale so axis labels remain meaningful.

    NaN and Inf values are silently excluded before histogramming
    (same behaviour as ``NMToolStats2.histogram``).

    Args:
        bins:    Number of equal-width bins (int) or explicit bin-edge
                 list.  Defaults to 10.
        x0:      Start of the xscale window (default ``-inf`` = beginning
                 of array).
        x1:      End of the xscale window (default ``+inf`` = end of array).
        xrange:  ``(min, max)`` tuple to restrict the amplitude range.
                 Defaults to None (full range of each array).
        density: If True, return probability density instead of counts.
                 Defaults to False.
    """

    name = "histogram"

    def __init__(
        self,
        bins: int | list = 10,
        x0: float = -math.inf,
        x1: float = math.inf,
        xrange: tuple | None = None,
        density: bool = False,
    ) -> None:
        self.bins = bins
        self.x0 = x0
        self.x1 = x1
        self.xrange = xrange
        self.density = density
        self._out_prefix: str = "H_"

    @property
    def out_prefix(self) -> str:
        """Prefix for output array names (default ``"H_"``)."""
        return self._out_prefix

    @out_prefix.setter
    def out_prefix(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "out_prefix", "string"))
        self._out_prefix = value
        self._results: dict = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, value) -> None:
        if isinstance(value, bool):
            raise TypeError("bins must be int or list, not bool")
        if isinstance(value, int):
            if value < 1:
                raise ValueError("bins must be >= 1, got %d" % value)
        elif isinstance(value, list):
            pass  # numpy validates content at np.histogram in run()
        else:
            raise TypeError(nmu.type_error_str(value, "bins", "int or list"))
        self._bins = value

    @property
    def x0(self) -> float:
        """Start of the xscale window (default ``-inf``)."""
        return self._x0

    @x0.setter
    def x0(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x0", "float"))
        if math.isnan(float(value)):
            raise ValueError("x0 must not be NaN")
        self._x0 = float(value)

    @property
    def x1(self) -> float:
        """End of the xscale window (default ``+inf``)."""
        return self._x1

    @x1.setter
    def x1(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "x1", "float"))
        if math.isnan(float(value)):
            raise ValueError("x1 must not be NaN")
        self._x1 = float(value)

    @property
    def xrange(self) -> tuple | None:
        return self._xrange

    @xrange.setter
    def xrange(self, value: tuple | None) -> None:
        if value is None:
            self._xrange = None
            return
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("xrange must be a 2-tuple (min, max) or None")
        self._xrange = value

    @property
    def density(self) -> bool:
        return self._density

    @density.setter
    def density(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "density", "bool"))
        self._density = value

    @property
    def results(self) -> dict:
        """Per-output-array dict; populated after :meth:`run_all`."""
        return self._results

    # ------------------------------------------------------------------
    # Validation helper

    def _validate_window(self) -> None:
        if self._x1 < self._x0:
            raise ValueError(
                "x1 (%g) must be >= x0 (%g)" % (self._x1, self._x0)
            )

    # ------------------------------------------------------------------
    # NMMainOp interface
    # ------------------------------------------------------------------

    def run_init(self) -> None:
        self._validate_window()
        self._results = {}

    def run(self, data: NMData, channel_name: str | None = None) -> None:
        if not isinstance(data.nparray, np.ndarray):
            return
        arr = data.nparray.astype(float)

        # Apply xscale window if either bound is finite
        if not (self._x0 == -math.inf and self._x1 == math.inf):
            sl = nm_math.xscale_window_to_slice(
                arr, data.xscale.to_dict(), self._x0, self._x1
            )
            arr = arr[sl]

        arr_finite = arr[np.isfinite(arr)]  # exclude NaN and Inf

        counts, edges = np.histogram(
            arr_finite, bins=self._bins,
            range=self._xrange, density=self._density,
        )

        base_name = self._out_prefix + data.name
        out_name = self._make_out_name(self._folder, base_name) if self._folder is not None else base_name
        if self._folder is not None:
            xscale = {
                "start": float(edges[0]),
                "delta": float(edges[1] - edges[0]),
                "label": data.yscale.label,  # input y-axis → output x-axis
                "units": data.yscale.units,
            }
            yscale = {
                "label": "density" if self._density else "counts",
                "units": "",
            }
            out_data = self._write_out_array(
                self._folder, out_name, counts.astype(float),
                xscale=xscale, yscale=yscale,
            )
            if out_data is not None:
                self._add_op_note(out_data, self._op_params_str())
        self._results[out_name] = {
            "counts": counts,
            "edges": edges,
            "n_excluded": data.nparray.size - len(arr_finite),
        }

    def _op_params_str(self) -> str:
        return "bins=%r, x0=%r, x1=%r, xrange=%r, density=%r" % (
            self._bins, self._x0, self._x1, self._xrange, self._density)


# =========================================================================
# Registry and lookup
# =========================================================================


_OP_REGISTRY: dict[str, type[NMMainOp]] = {
    # --- accumulation (combine multiple arrays) ---
    "average": NMMainOpAverage,
    "concatenate": NMMainOpConcatenate,
    "sum": NMMainOpSum,
    "sum_sqr": NMMainOpSumSqr,
    # --- arithmetic / scaling (element-wise value operations) ---
    "arithmetic": NMMainOpArithmetic,
    "arithmetic_by_array": NMMainOpArithmeticByArray,
    "baseline": NMMainOpBaseline,
    "dfof": NMMainOpDFOF,
    "normalize": NMMainOpNormalize,
    "rescale": NMMainOpRescale,
    # --- calculus / signal conditioning ---
    "differentiate": NMMainOpDifferentiate,
    "filter": NMMainOpFilter,
    "integrate": NMMainOpIntegrate,
    "smooth": NMMainOpSmooth,
    # --- statistics / comparison ---
    "histogram": NMMainOpHistogram,
    "inequality": NMMainOpInequality,
    "max": NMMainOpMax,
    "min": NMMainOpMin,
    # --- array structure ---
    "delete_nans": NMMainOpDeleteNaNs,
    "delete_points": NMMainOpDeletePoints,
    "insert_points": NMMainOpInsertPoints,
    "redimension": NMMainOpRedimension,
    "replace_values": NMMainOpReplaceValues,
    "reverse": NMMainOpReverse,
    "rotate": NMMainOpRotate,
    # --- x-axis ---
    "align": NMMainOpAlign,
    "interpolate": NMMainOpInterpolate,
    "resample": NMMainOpResample,
    "rescale_x": NMMainOpRescaleX,
}


def op_from_name(name: str) -> NMMainOp:
    """Instantiate an NMMainOp subclass by name.

    Args:
        name: Case-insensitive op name (e.g. ``"average"``, ``"scale"``).

    Returns:
        A new NMMainOp instance with default parameters.

    Raises:
        TypeError: If name is not a string.
        ValueError: If name is not in the registry.
    """
    if not isinstance(name, str):
        raise TypeError(nmu.type_error_str(name, "name", "string"))
    cls = _OP_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            "unknown op: '%s'; valid ops: %s" % (name, sorted(_OP_REGISTRY))
        )
    return cls()
