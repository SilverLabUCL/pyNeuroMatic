# -*- coding: utf-8 -*-
"""
NMMainOp — operation classes for NMToolMain.

Provides a base class NMMainOp and concrete subclasses (NMMainOpAverage,
NMMainOpScale) following the same pattern as nm_transform.py:
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
        """
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
# Average
# =========================================================================


class NMMainOpAverage(NMMainOp):
    """Average selected data waves per channel.

    Accumulates arrays by channel, truncates all arrays to the shortest
    length, and writes the mean as a new NMData wave
    ``Avg_{prefix}{channel}`` (e.g. ``Avg_RecordA``) into the source folder.

    Parameters:
        ignore_nans: If True (default) use ``np.nanmean``; otherwise
            ``np.mean`` (NaN propagates to the result).
    """

    name = "average"

    def __init__(self, ignore_nans: bool = True) -> None:
        if not isinstance(ignore_nans, bool):
            raise TypeError(
                nmu.type_error_str(ignore_nans, "ignore_nans", "boolean")
            )
        self._ignore_nans = ignore_nans
        self._results: dict[str, str] = {}  # channel → output name

    @property
    def ignore_nans(self) -> bool:
        """If True, NaN values are excluded from the mean (np.nanmean)."""
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

    def run_finish(
        self,
        folder: NMFolder | None = None,
        prefix: str | None = None,
    ) -> None:
        """Compute the mean and write one output wave per channel to folder.

        Args:
            folder: Destination NMFolder for the averaged waves.
            prefix: Dataseries name used as the output wave prefix.  Falls
                back to the prefix parsed from the first wave name if None.
        """
        if not self._accum or folder is None:
            return

        pfx = prefix if prefix is not None else (self._parsed_prefix or "")
        for cname, arrays in self._accum.items():
            min_len = min(len(a) for a in arrays)
            stack = np.stack([a[:min_len] for a in arrays])
            avg = np.nanmean(stack, axis=0) if self._ignore_nans else np.mean(stack, axis=0)
            out_name = "Avg_" + pfx + cname
            folder.data.new(
                out_name,
                nparray=avg,
                xscale=self._xscales[cname],
                yscale=self._yscales[cname],
            )
            out_data = folder.data.get(out_name)
            if out_data is not None:
                n = len(arrays)
                ds_name = prefix if prefix is not None else (self._parsed_prefix or "")
                epoch_nums = []
                for dname in self._data_names.get(cname, []):
                    parsed = nmu.parse_data_name(dname)
                    epoch_nums.append(str(parsed[2]) if parsed is not None else dname)
                try:
                    ints = [int(e) for e in epoch_nums]
                    lo, hi = min(ints), max(ints)
                    if len(ints) > 1 and ints == list(range(lo, hi + 1)):
                        epoch_str = "%d-%d" % (lo, hi)
                    else:
                        epoch_str = "[" + ",".join(epoch_nums) + "]"
                except ValueError:
                    epoch_str = "[" + ",".join(epoch_nums) + "]"
                self._add_note(
                    out_data,
                    "NMAverage(folder=%s,dataseries=%s,channel=%s,epochs=%s,n_epochs=%d)"
                    % (folder.name, ds_name, cname, epoch_str, n),
                )
            self._results[cname] = out_name


# =========================================================================
# Scale
# =========================================================================


class NMMainOpScale(NMMainOp):
    """Multiply each selected wave by a scalar factor (in-place).

    Parameters:
        factor: Multiplication factor (default 1.0).  Must be int or float
            (not bool).
    """

    name = "scale"

    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor  # use setter for validation

    @property
    def factor(self) -> float:
        """Multiplication factor applied to each wave."""
        return self._factor

    @factor.setter
    def factor(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "factor", "float"))
        self._factor = float(value)

    def run(
        self,
        data: NMData,
        channel_name: str | None = None,
    ) -> None:
        """Multiply data.nparray by self.factor in-place.

        Args:
            data: The NMData object to scale.
            channel_name: Unused; present for API consistency.
        """
        if not isinstance(data.nparray, np.ndarray):
            return
        data.nparray = data.nparray * self._factor
        self._add_note(data, "NMScale(factor=%.6g)" % self._factor)


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
        In ``average`` mode the mean of all per-wave baselines for each channel
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
# Registry and lookup
# =========================================================================


_OP_REGISTRY: dict[str, type[NMMainOp]] = {
    "average": NMMainOpAverage,
    "baseline": NMMainOpBaseline,
    "delete_points": NMMainOpDeletePoints,
    "differentiate": NMMainOpDifferentiate,
    "insert_points": NMMainOpInsertPoints,
    "integrate": NMMainOpIntegrate,
    "redimension": NMMainOpRedimension,
    "reverse": NMMainOpReverse,
    "rotate": NMMainOpRotate,
    "scale": NMMainOpScale,
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
