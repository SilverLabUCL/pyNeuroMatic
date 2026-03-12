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

import numpy as np

from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
import pyneuromatic.core.nm_utilities as nmu


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
    """

    name: str = ""

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
            prefix: Dataseries name to use as the output wave prefix.  If
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


# =========================================================================
# Accumulate (base for Average, Sum, SumSqr, Min, Max)
# =========================================================================


class NMMainOpAccumulate(NMMainOp):
    """Base class for ops that accumulate waves and reduce them per channel.

    Subclasses set two class attributes and override ``_reduce()``:

    - ``_output_prefix``: prefix for the output wave name (e.g. ``"Avg_"``).
    - ``_note_name``: op name used in the note string (e.g. ``"NMAverage"``).
    - ``_reduce(stack)``: compute the per-channel result from the stacked array.

    The full lifecycle is handled here: ``run_init`` resets buffers,
    ``run`` accumulates per-channel arrays, and ``run_finish`` reduces
    and writes one output wave per channel.

    Parameters:
        ignore_nans: If True (default) use NaN-aware reductions
            (``np.nanmean``, ``np.nansum``, etc.); otherwise NaN propagates.
    """

    _output_prefix: str = ""
    _note_name: str = ""

    def __init__(self, ignore_nans: bool = True) -> None:
        if not isinstance(ignore_nans, bool):
            raise TypeError(
                nmu.type_error_str(ignore_nans, "ignore_nans", "boolean")
            )
        self._ignore_nans = ignore_nans
        self._results: dict[str, str] = {}  # channel → output name

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
    def results(self) -> dict[str, str]:
        """Read-only dict mapping channel name → output NMData name."""
        return dict(self._results)

    def _reduce(self, stack: np.ndarray) -> np.ndarray:
        """Reduce the stacked array to a single output array.

        Args:
            stack: 2-D array of shape (n_waves, n_points).

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
        """Accumulate one wave into the per-channel buffer.

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
        note_name: str,
        folder_name: str,
        ds_name: str,
        cname: str,
        epoch_str: str,
        n: int,
    ) -> str:
        """Build the note string for an output wave."""
        return (
            "%s(folder=%s,dataseries=%s,channel=%s,epochs=%s,n_epochs=%d)"
            % (note_name, folder_name, ds_name, cname, epoch_str, n)
        )

    def _write_output_wave(
        self,
        folder: NMFolder,
        out_name: str,
        arr: np.ndarray,
        note_name: str,
        folder_name: str,
        ds_name: str,
        cname: str,
        epoch_str: str,
        n: int,
    ) -> None:
        """Create an output wave in folder and add a note to it."""
        folder.data.new(
            out_name,
            nparray=arr,
            xscale=self._xscales[cname],
            yscale=self._yscales[cname],
        )
        out_data = folder.data.get(out_name)
        if out_data is not None:
            self._add_note(
                out_data,
                self._make_note_str(note_name, folder_name, ds_name, cname, epoch_str, n),
            )

    def _process_channel(
        self,
        folder: NMFolder,
        pfx: str,
        cname: str,
        arrays: list[np.ndarray],
    ) -> None:
        """Reduce and write one output wave for a single channel.

        Subclasses may override to write additional output waves (e.g.
        Stdv, Var, SEM alongside the mean).

        Args:
            folder: Destination NMFolder.
            pfx: Dataseries prefix for the output wave name.
            cname: Channel name.
            arrays: List of accumulated arrays for this channel.
        """
        min_len = min(len(a) for a in arrays)
        stack = np.stack([a[:min_len] for a in arrays])
        n = len(arrays)
        epoch_str = self._epoch_str(cname)
        arr = self._reduce(stack)
        out_name = self._output_prefix + pfx + cname
        self._write_output_wave(
            folder, out_name, arr, self._note_name, folder.name, pfx, cname, epoch_str, n
        )
        self._results[cname] = out_name

    def run_finish(
        self,
        folder: NMFolder | None = None,
        prefix: str | None = None,
    ) -> None:
        """Reduce and write one output wave per channel to folder.

        Args:
            folder: Destination NMFolder for the output waves.
            prefix: Dataseries name used as the output wave prefix.  Falls
                back to the prefix parsed from the first wave name if None.
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
    """Average selected data waves per channel.

    Accumulates arrays by channel, truncates all arrays to the shortest
    length, and writes the mean as ``Avg_{prefix}{channel}``
    (e.g. ``Avg_RecordA``) into the source folder.

    Optionally also writes Stdv, Var, and/or SEM waves (sample statistics,
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
    _note_name = "NMAverage"

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
        """If True, write a Stdv output wave alongside the mean."""
        return self._compute_stdv

    @compute_stdv.setter
    def compute_stdv(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "compute_stdv", "boolean"))
        self._compute_stdv = value

    @property
    def compute_var(self) -> bool:
        """If True, write a Var output wave alongside the mean."""
        return self._compute_var

    @compute_var.setter
    def compute_var(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "compute_var", "boolean"))
        self._compute_var = value

    @property
    def compute_sem(self) -> bool:
        """If True, write a SEM output wave alongside the mean."""
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
        """Write mean wave, then optionally Stdv/Var/SEM waves."""
        super()._process_channel(folder, pfx, cname, arrays)
        if not (self._compute_stdv or self._compute_var or self._compute_sem):
            return
        min_len = min(len(a) for a in arrays)
        stack = np.stack([a[:min_len] for a in arrays])
        n = len(arrays)
        epoch_str = self._epoch_str(cname)
        std_fn = np.nanstd if self._ignore_nans else np.std
        std = std_fn(stack, axis=0, ddof=1)
        if self._compute_stdv:
            self._write_output_wave(
                folder, "Stdv_" + pfx + cname, std,
                "NMStdv", folder.name, pfx, cname, epoch_str, n,
            )
        if self._compute_var:
            self._write_output_wave(
                folder, "Var_" + pfx + cname, std ** 2,
                "NMVar", folder.name, pfx, cname, epoch_str, n,
            )
        if self._compute_sem:
            self._write_output_wave(
                folder, "SEM_" + pfx + cname, std / np.sqrt(n),
                "NMSEM", folder.name, pfx, cname, epoch_str, n,
            )


# =========================================================================
# Sum, SumSqr, Min, Max  (NMMainOpAccumulate subclasses)
# =========================================================================


class NMMainOpSum(NMMainOpAccumulate):
    """Sum selected data waves point-by-point per channel.

    Writes ``Sum_{prefix}{channel}`` (e.g. ``Sum_RecordA``) to the folder.

    Parameters:
        ignore_nans: If True (default) use ``np.nansum`` (NaN treated as 0);
            otherwise ``np.sum`` (NaN propagates).
    """

    name = "sum"
    _output_prefix = "Sum_"
    _note_name = "NMSum"

    def _reduce(self, stack: np.ndarray) -> np.ndarray:
        return np.nansum(stack, axis=0) if self._ignore_nans else np.sum(stack, axis=0)


class NMMainOpSumSqr(NMMainOpAccumulate):
    """Sum of squares of selected data waves point-by-point per channel.

    Squares each wave then sums, writing ``SumSqr_{prefix}{channel}``
    (e.g. ``SumSqr_RecordA``) to the folder.

    Parameters:
        ignore_nans: If True (default) use ``np.nansum`` on the squared array;
            otherwise ``np.sum`` (NaN propagates).
    """

    name = "sum_sqr"
    _output_prefix = "SumSqr_"
    _note_name = "NMSumSqr"

    def _reduce(self, stack: np.ndarray) -> np.ndarray:
        sq = stack ** 2
        return np.nansum(sq, axis=0) if self._ignore_nans else np.sum(sq, axis=0)


class NMMainOpMin(NMMainOpAccumulate):
    """Point-by-point minimum across selected data waves per channel.

    Writes ``Min_{prefix}{channel}`` (e.g. ``Min_RecordA``) to the folder.

    Parameters:
        ignore_nans: If True (default) use ``np.nanmin`` (NaN ignored);
            otherwise ``np.min`` (NaN propagates).
    """

    name = "min"
    _output_prefix = "Min_"
    _note_name = "NMMin"

    def _reduce(self, stack: np.ndarray) -> np.ndarray:
        return np.nanmin(stack, axis=0) if self._ignore_nans else np.min(stack, axis=0)


class NMMainOpMax(NMMainOpAccumulate):
    """Point-by-point maximum across selected data waves per channel.

    Writes ``Max_{prefix}{channel}`` (e.g. ``Max_RecordA``) to the folder.

    Parameters:
        ignore_nans: If True (default) use ``np.nanmax`` (NaN ignored);
            otherwise ``np.max`` (NaN propagates).
    """

    name = "max"
    _output_prefix = "Max_"
    _note_name = "NMMax"

    def _reduce(self, stack: np.ndarray) -> np.ndarray:
        return np.nanmax(stack, axis=0) if self._ignore_nans else np.max(stack, axis=0)


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
        if value not in _VALID_ARITH_OPS:
            raise ValueError(
                "op must be one of %s, got %r" % (sorted(_VALID_ARITH_OPS), value)
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
        data.nparray = _apply_op(data.nparray.astype(float), factor, self._op)
        self._add_note(
            data, "NMArithmetic(factor=%.6g,op=%s)" % (factor, self._op)
        )


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
            name that will be looked up in the source folder at run time.
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
        if value not in _VALID_ARITH_OPS:
            raise ValueError(
                "op must be one of %s, got %r" % (sorted(_VALID_ARITH_OPS), value)
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
        data.nparray = _apply_op(arr, self._resolved_ref, self._op)
        ref_label = self._ref if isinstance(self._ref, str) else "array"
        self._add_note(
            data, "NMArithmeticByArray(ref=%s,op=%s)" % (ref_label, self._op)
        )


# =========================================================================
# Redimension
# =========================================================================


class NMMainOpRedimension(NMMainOp):
    """Change the number of points in each selected wave (in-place).

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
        """Pad value used when extending a wave."""
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
            self._add_note(data, "NMRedimension(n_points=%d)" % n)
        else:
            data.nparray = np.concatenate([arr, np.full(n - old_len, self._fill)])
            self._add_note(data, "NMRedimension(n_points=%d,fill=%.6g)" % (n, self._fill))


# =========================================================================
# Insert points
# =========================================================================


class NMMainOpInsertPoints(NMMainOp):
    """Insert points into each selected wave at a given index (in-place).

    Points at and after ``index`` are shifted right.  Equivalent to Igor's
    ``InsertPoints pos, n, wave``.

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
        self._add_note(
            data,
            "NMInsertPoints(index=%d,n_points=%d,fill=%.6g)"
            % (self._index, self._n_points, self._fill),
        )


# =========================================================================
# Delete points
# =========================================================================


class NMMainOpDeletePoints(NMMainOp):
    """Delete points from each selected wave at a given index (in-place).

    Equivalent to Igor's ``DeletePoints pos, n, wave``.

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
        self._add_note(
            data,
            "NMDeletePoints(index=%d,n_points=%d)" % (self._index, self._n_points),
        )


# =========================================================================
# Baseline helper
# =========================================================================


def _time_to_slice(arr: np.ndarray, xscale_dict: dict, t_begin: float, t_end: float) -> slice:
    """Convert a time window to an array slice using xscale start/delta.

    Clips to valid range; returns an empty slice if the window is fully outside.
    """
    start = xscale_dict.get("start", 0.0)
    delta = xscale_dict.get("delta", 1.0)
    if delta == 0:
        return slice(0, 0)
    i0 = int(round((t_begin - start) / delta))
    i1 = int(round((t_end - start) / delta)) + 1  # inclusive end
    i0 = max(0, i0)
    i1 = min(len(arr), i1)
    return slice(i0, i1)


def _compute_ref(arr: np.ndarray, fxn: str, n_mean: int) -> float:
    """Compute a scalar reference value from arr using the given function.

    Args:
        arr: Array slice to compute the reference from.
        fxn: One of ``"mean"``, ``"min"``, ``"max"``, ``"mean@min"``,
            ``"mean@max"``.
        n_mean: Number of points to compute mean around the extremum (used
            only for ``"mean@min"`` and ``"mean@max"``).

    Returns:
        Scalar reference value, or ``float("nan")`` if arr is empty.
    """
    if len(arr) == 0:
        return float("nan")
    if fxn == "mean":
        return float(np.nanmean(arr))
    elif fxn == "min":
        return float(np.nanmin(arr))
    elif fxn == "max":
        return float(np.nanmax(arr))
    elif fxn == "mean@min":
        i = int(np.nanargmin(arr))
        half = n_mean // 2
        return float(np.nanmean(arr[max(0, i - half):i + half + 1]))
    elif fxn == "mean@max":
        i = int(np.nanargmax(arr))
        half = n_mean // 2
        return float(np.nanmean(arr[max(0, i - half):i + half + 1]))
    return float("nan")


_VALID_ARITH_OPS: frozenset[str] = frozenset({"x", "/", "+", "-", "=", "**"})


def _apply_op(arr: np.ndarray, value, op: str) -> np.ndarray:
    """Apply a binary operation between arr and value.

    Args:
        arr: Input array.
        value: Scalar float or ndarray of the same length as arr.
        op: One of ``"x"`` (multiply), ``"/"`` (divide), ``"+"`` (add),
            ``"-"`` (subtract), ``"="`` (assign constant/array),
            ``"**"`` (exponentiate).

    Returns:
        Result array (same dtype promotion as numpy default, except ``"="``
        which returns float64).
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
# Baseline
# =========================================================================


class NMMainOpBaseline(NMMainOp):
    """Subtract a baseline from each selected wave.

    Two modes are supported:

    - **per_wave**: Each wave's own baseline (mean of the window) is subtracted
      from that wave independently.
    - **average**: A single shared baseline per channel is computed as the mean
      of all per-wave baselines for that channel, then subtracted from every
      wave in that channel.

    Parameters:
        t_begin: Baseline window start in time units (default 0.0).
        t_end: Baseline window end in time units (default 0.0).  Must be >=
            ``t_begin``.
        mode: ``"per_wave"`` (default) or ``"average"``.
        ignore_nans: If True (default) use ``np.nanmean``; otherwise ``np.mean``
            (NaN propagates to the result).
    """

    name = "baseline"

    _VALID_MODES = {"per_wave", "average"}

    def __init__(
        self,
        t_begin: float = 0.0,
        t_end: float = 0.0,
        mode: str = "per_wave",
        ignore_nans: bool = True,
    ) -> None:
        self.t_begin = t_begin
        self.t_end = t_end
        self.mode = mode
        self.ignore_nans = ignore_nans

    # ------------------------------------------------------------------
    # Properties

    @property
    def t_begin(self) -> float:
        """Baseline window start (time units)."""
        return self._t_begin

    @t_begin.setter
    def t_begin(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "t_begin", "float"))
        self._t_begin = float(value)

    @property
    def t_end(self) -> float:
        """Baseline window end (time units)."""
        return self._t_end

    @t_end.setter
    def t_end(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "t_end", "float"))
        self._t_end = float(value)

    @property
    def mode(self) -> str:
        """Subtraction mode: ``'per_wave'`` or ``'average'``."""
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
        if self._t_end < self._t_begin:
            raise ValueError(
                "t_end (%g) must be >= t_begin (%g)" % (self._t_end, self._t_begin)
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
        """Compute and (optionally) apply baseline for one wave.

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

        sl = _time_to_slice(
            data.nparray, data.xscale.to_dict(), self._t_begin, self._t_end
        )
        segment = data.nparray[sl].astype(float)
        if len(segment) == 0:
            baseline = 0.0
        elif self._ignore_nans:
            baseline = float(np.nanmean(segment))
        else:
            baseline = float(np.mean(segment))

        if self._mode == "per_wave":
            data.nparray = data.nparray.astype(float) - baseline
            self._add_note(
                data,
                "NMBaseline(t_begin=%.6g,t_end=%.6g,mode=per_wave,baseline=%.6g)"
                % (self._t_begin, self._t_end, baseline),
            )
        else:  # "average"
            self._baseline_accum.setdefault(channel_name, []).append(baseline)
            self._data_refs.setdefault(channel_name, []).append(data)

    def run_finish(
        self,
        folder: NMFolder | None = None,
        prefix: str | None = None,
    ) -> None:
        """Apply averaged baseline (average mode only).

        In ``per_wave`` mode this is a no-op (subtraction was done in ``run()``).
        In ``average`` mode the average of all per-wave baselines for each channel
        is computed and subtracted from every wave in that channel.
        """
        if self._mode == "per_wave":
            return
        for channel_name, baselines in self._baseline_accum.items():
            avg_baseline = float(
                np.nanmean(baselines) if self._ignore_nans else np.mean(baselines)
            )
            for d in self._data_refs[channel_name]:
                d.nparray = d.nparray.astype(float) - avg_baseline
                self._add_note(
                    d,
                    "NMBaseline(t_begin=%.6g,t_end=%.6g,mode=average,channel=%s,baseline=%.6g)"
                    % (self._t_begin, self._t_end, channel_name, avg_baseline),
                )


# =========================================================================
# Reverse
# =========================================================================


class NMMainOpReverse(NMMainOp):
    """Reverse each selected wave in-place.

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
        self._add_note(data, "NMReverse()")


# =========================================================================
# Rotate
# =========================================================================


class NMMainOpRotate(NMMainOp):
    """Rotate each selected wave by ``n_points`` positions (in-place).

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
        self._add_note(data, "NMRotate(n_points=%d)" % self._n_points)


# =========================================================================
# Integrate
# =========================================================================


class NMMainOpIntegrate(NMMainOp):
    """Cumulative integration of each selected wave (in-place).

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
        self._add_note(data, "NMIntegrate(method=%s)" % self._method)


# =========================================================================
# Differentiate
# =========================================================================


class NMMainOpDifferentiate(NMMainOp):
    """First derivative of each selected wave using ``np.gradient`` (in-place).

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
        self._add_note(data, "NMDifferentiate()")


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
        self._add_note(
            data,
            "NMReplaceValues(old=%.6g,new=%.6g,n=%d)"
            % (self._old_value, self._new_value, n),
        )


# =========================================================================
# Delete NaNs
# =========================================================================


class NMMainOpDeleteNaNs(NMMainOp):
    """Remove NaN and/or ±Inf points from each selected wave (in-place).

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
        self._add_note(
            data,
            "NMDeleteNaNs(delete_nans=%s,delete_infs=%s,n=%d)"
            % (self._delete_nans, self._delete_infs, n),
        )


# =========================================================================
# Normalize
# =========================================================================


class NMMainOpNormalize(NMMainOp):
    """Rescale each wave so a low reference maps to norm_min and a high reference maps to norm_max.

    Two independent time windows are used:

    - Window 1 (``t_begin1``/``t_end1``) computes the "low" reference via
      ``fxn1`` (``"mean"``, ``"min"``, or ``"mean@min"``).
    - Window 2 (``t_begin2``/``t_end2``) computes the "high" reference via
      ``fxn2`` (``"mean"``, ``"max"``, or ``"mean@max"``).

    Two modes (matching NMMainOpBaseline):

    - **per_wave**: Each wave is normalized to its own references.
    - **average**: Per-channel mean references are computed across all waves,
      then applied to every wave in that channel.

    Parameters:
        t_begin1: Window 1 start in time units (default 0.0).
        t_end1: Window 1 end in time units (default 0.0).
        fxn1: Function for the low reference: ``"mean"``, ``"min"``, or
            ``"mean@min"`` (default ``"mean"``).
        n_mean1: Points around min for ``mean@min`` (default 1).
        t_begin2: Window 2 start in time units (default 0.0).
        t_end2: Window 2 end in time units (default 0.0).
        fxn2: Function for the high reference: ``"mean"``, ``"max"``, or
            ``"mean@max"`` (default ``"mean"``).
        n_mean2: Points around max for ``mean@max`` (default 1).
        norm_min: Target normalized minimum (default 0.0).
        norm_max: Target normalized maximum (default 1.0).
        mode: ``"per_wave"`` (default) or ``"average"``.
    """

    name = "normalize"

    _VALID_FXN1 = {"mean", "min", "mean@min"}
    _VALID_FXN2 = {"mean", "max", "mean@max"}
    _VALID_MODES = {"per_wave", "average"}

    def __init__(
        self,
        t_begin1: float = 0.0,
        t_end1: float = 0.0,
        fxn1: str = "mean",
        n_mean1: int = 1,
        t_begin2: float = 0.0,
        t_end2: float = 0.0,
        fxn2: str = "mean",
        n_mean2: int = 1,
        norm_min: float = 0.0,
        norm_max: float = 1.0,
        mode: str = "per_wave",
    ) -> None:
        self.t_begin1 = t_begin1
        self.t_end1 = t_end1
        self.fxn1 = fxn1
        self.n_mean1 = n_mean1
        self.t_begin2 = t_begin2
        self.t_end2 = t_end2
        self.fxn2 = fxn2
        self.n_mean2 = n_mean2
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.mode = mode

    # ------------------------------------------------------------------
    # Properties

    @property
    def t_begin1(self) -> float:
        """Window 1 start (time units)."""
        return self._t_begin1

    @t_begin1.setter
    def t_begin1(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "t_begin1", "float"))
        self._t_begin1 = float(value)

    @property
    def t_end1(self) -> float:
        """Window 1 end (time units)."""
        return self._t_end1

    @t_end1.setter
    def t_end1(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "t_end1", "float"))
        self._t_end1 = float(value)

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
    def t_begin2(self) -> float:
        """Window 2 start (time units)."""
        return self._t_begin2

    @t_begin2.setter
    def t_begin2(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "t_begin2", "float"))
        self._t_begin2 = float(value)

    @property
    def t_end2(self) -> float:
        """Window 2 end (time units)."""
        return self._t_end2

    @t_end2.setter
    def t_end2(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "t_end2", "float"))
        self._t_end2 = float(value)

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
        """Normalization mode: ``'per_wave'`` or ``'average'``."""
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
        if self._t_end1 < self._t_begin1:
            raise ValueError(
                "t_end1 (%g) must be >= t_begin1 (%g)" % (self._t_end1, self._t_begin1)
            )
        if self._t_end2 < self._t_begin2:
            raise ValueError(
                "t_end2 (%g) must be >= t_begin2 (%g)" % (self._t_end2, self._t_begin2)
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
        note = (
            "NMNormalize(t_begin1=%.6g,t_end1=%.6g,fxn1=%s,"
            "t_begin2=%.6g,t_end2=%.6g,fxn2=%s,"
            "norm_min=%.6g,norm_max=%.6g,mode=%s"
        ) % (
            self._t_begin1, self._t_end1, self._fxn1,
            self._t_begin2, self._t_end2, self._fxn2,
            self._norm_min, self._norm_max, self._mode,
        )
        if channel_name is not None:
            note += ",channel=%s" % channel_name
        note += ",ref_min=%.6g,ref_max=%.6g)" % (ref_min, ref_max)
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
        """Compute references and (optionally) normalize one wave.

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
        sl1 = _time_to_slice(arr, xd, self._t_begin1, self._t_end1)
        sl2 = _time_to_slice(arr, xd, self._t_begin2, self._t_end2)
        ref_min = _compute_ref(arr[sl1], self._fxn1, self._n_mean1)
        ref_max = _compute_ref(arr[sl2], self._fxn2, self._n_mean2)

        if self._mode == "per_wave":
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

        In ``per_wave`` mode this is a no-op (normalization was done in
        ``run()``).  In ``average`` mode the mean of all per-wave references
        for each channel is computed and applied to every wave in that channel.
        """
        if self._mode == "per_wave":
            return
        for channel_name, ref_mins in self._ref_min_accum.items():
            avg_ref_min = float(np.nanmean(ref_mins))
            avg_ref_max = float(np.nanmean(self._ref_max_accum[channel_name]))
            for d in self._data_refs[channel_name]:
                arr = d.nparray.astype(float)
                d.nparray = self._apply(arr, avg_ref_min, avg_ref_max)
                self._add_note(d, self._note_str(avg_ref_min, avg_ref_max, channel_name))


# =========================================================================
# Registry and lookup
# =========================================================================


_OP_REGISTRY: dict[str, type[NMMainOp]] = {
    "arithmetic": NMMainOpArithmetic,
    "arithmetic_by_array": NMMainOpArithmeticByArray,
    "average": NMMainOpAverage,
    "baseline": NMMainOpBaseline,
    "delete_nans": NMMainOpDeleteNaNs,
    "delete_points": NMMainOpDeletePoints,
    "differentiate": NMMainOpDifferentiate,
    "insert_points": NMMainOpInsertPoints,
    "integrate": NMMainOpIntegrate,
    "max": NMMainOpMax,
    "min": NMMainOpMin,
    "normalize": NMMainOpNormalize,
    "redimension": NMMainOpRedimension,
    "replace_values": NMMainOpReplaceValues,
    "reverse": NMMainOpReverse,
    "rotate": NMMainOpRotate,
    "sum": NMMainOpSum,
    "sum_sqr": NMMainOpSumSqr,
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
