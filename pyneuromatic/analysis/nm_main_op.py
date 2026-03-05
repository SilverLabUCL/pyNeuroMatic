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
    lifecycle suitable for pointwise operations (e.g. Scale).
    Aggregating operations (e.g. Average) override ``run_all()`` directly
    to receive the complete data list at once.

    Subclasses should set the class attribute ``name`` to a short lowercase
    string matching the registry key (e.g. ``"scale"``).
    """

    name: str = ""

    def run_all(
        self,
        data_items: list[tuple[NMData, str | None, str | None]],
        folder: NMFolder | None,
    ) -> None:
        """Process all data items.

        Default implementation calls ``run_init()``, then ``run()`` for each
        item, then ``run_finish()``.  Aggregating ops (e.g. Average) override
        this method instead of ``run()``.

        Args:
            data_items: List of ``(NMData, channel_name, prefix)`` triples.
                channel_name and prefix may be ``None`` when running in
                direct-data mode (no dataseries context); ops then fall back
                to parsing these from the data name.
            folder: The NMFolder that owns the source data.  Passed to
                ``run_finish()`` so ops can write output there.
        """
        self.run_init()
        for data, channel_name, prefix in data_items:
            self.run(data, channel_name)
        self.run_finish(folder)

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
            NotImplementedError: If the subclass does not override this method
                and does not override ``run_all()``.
        """
        raise NotImplementedError(
            "%s.run() not implemented" % self.__class__.__name__
        )

    def run_finish(self, folder: NMFolder | None) -> None:
        """Called once after the per-item loop.  Override to write results."""


# =========================================================================
# Average
# =========================================================================


class NMMainOpAverage(NMMainOp):
    """Average selected data waves per channel.

    Accumulates arrays by channel across the data_items list, truncates all
    arrays to the shortest length, and writes the mean as a new NMData wave
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

    def run_all(
        self,
        data_items: list[tuple[NMData, str | None, str | None]],
        folder: NMFolder | None,
    ) -> None:
        """Average all data items per channel and write results to folder.

        Args:
            data_items: List of ``(NMData, channel_name, prefix)`` triples.
            folder: Destination NMFolder for the averaged waves.
        """
        self._results.clear()

        # Phase 1: accumulate per channel
        accum: dict[str, list[np.ndarray]] = {}
        xscales: dict[str, dict] = {}
        yscales: dict[str, dict] = {}
        prefix: str | None = None

        for data, channel_name, item_prefix in data_items:
            if not isinstance(data.nparray, np.ndarray):
                continue

            # Determine channel
            if channel_name is None:
                parsed = nmu.parse_data_name(data.name)
                channel_name = parsed[1] if parsed is not None else "A"

            # Capture prefix: use dataseries name if provided, else parse from
            # first wave name (direct-data mode fallback)
            if prefix is None:
                if item_prefix is not None:
                    prefix = item_prefix
                else:
                    parsed = nmu.parse_data_name(data.name)
                    prefix = parsed[0] if parsed is not None else ""

            # First encounter for this channel: record scale metadata
            if channel_name not in accum:
                accum[channel_name] = []
                xscales[channel_name] = data.xscale.to_dict()
                yscales[channel_name] = data.yscale.to_dict()

            accum[channel_name].append(data.nparray.astype(float).copy())

        if not accum or folder is None:
            return

        # Phase 2: compute mean and save per channel
        pfx = prefix if prefix is not None else ""
        for cname, arrays in accum.items():
            min_len = min(len(a) for a in arrays)
            stack = np.stack([a[:min_len] for a in arrays])
            if self._ignore_nans:
                avg = np.nanmean(stack, axis=0)
            else:
                avg = np.mean(stack, axis=0)

            out_name = "Avg_" + pfx + cname
            folder.data.new(
                out_name,
                nparray=avg,
                xscale=xscales[cname],
                yscale=yscales[cname],
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


# =========================================================================
# Registry and lookup
# =========================================================================


_OP_REGISTRY: dict[str, type[NMMainOp]] = {
    "average": NMMainOpAverage,
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
