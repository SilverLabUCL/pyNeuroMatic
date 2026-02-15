# -*- coding: utf-8 -*-
"""
NMScale module — lightweight wrappers for x/y scale metadata.

Provides NMScaleY (label, units) and NMScaleX (label, units, start, delta)
with property-based access, validation, and history logging.

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

import copy
import math

import pyneuromatic.core.nm_history as nmh
import pyneuromatic.core.nm_preferences as nmp
import pyneuromatic.core.nm_utilities as nmu


class NMScaleY:
    """Y-scale metadata: label and units.

    Provides property-based access with validation and optional history
    logging. Used by NMData and NMChannel for y-axis scale parameters.
    """

    _path_suffix: str = "yscale"

    def __init__(
        self,
        parent: object | None = None,
        label: str = "",
        units: str = "",
    ) -> None:
        self._parent = parent
        self._label: str = str(label) if label else ""
        self._units: str = str(units) if units else ""

    def __repr__(self) -> str:
        return "NMScaleY(label='%s', units='%s')" % (self._label, self._units)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NMScaleY):
            return (
                self._label == other._label
                and self._units == other._units
            )
        if isinstance(other, dict):
            return self.to_dict() == other
        return NotImplemented

    def __deepcopy__(self, memo: dict) -> NMScaleY:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for attr, value in self.__dict__.items():
            if attr == "_parent":
                setattr(result, attr, None)
            else:
                setattr(result, attr, copy.deepcopy(value, memo))
        return result

    def __getitem__(self, key: str) -> object:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    @property
    def path_str(self) -> str:
        if self._parent is not None and hasattr(self._parent, "path_str"):
            return self._parent.path_str + "." + self._path_suffix
        return self._path_suffix

    # --- label ---

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._set_label(value)

    def _set_label(
        self,
        value: str,
        log: bool = True,
        quiet: bool = nmp.QUIET,
    ) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "label", "string"))
        value = value.strip()
        if value == self._label:
            return
        self._label = value
        if log:
            nmh.history(
                "set label='%s'" % value,
                path=self.path_str,
                quiet=quiet,
            )

    # --- units ---

    @property
    def units(self) -> str:
        return self._units

    @units.setter
    def units(self, value: str) -> None:
        self._set_units(value)

    def _set_units(
        self,
        value: str,
        log: bool = True,
        quiet: bool = nmp.QUIET,
    ) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "units", "string"))
        value = value.strip()
        if value == self._units:
            return
        self._units = value
        if log:
            nmh.history(
                "set units='%s'" % value,
                path=self.path_str,
                quiet=quiet,
            )

    # --- serialization ---

    def to_dict(self) -> dict:
        return {"label": self._label, "units": self._units}


class NMScaleX(NMScaleY):
    """X-scale metadata: label, units, start, and delta.

    Extends NMScaleY with start (x-axis origin) and delta (sample interval).
    Used by NMData and NMChannel for x-axis scale parameters.
    """

    _path_suffix: str = "xscale"

    def __init__(
        self,
        parent: object | None = None,
        label: str = "",
        units: str = "",
        start: float | int = 0,
        delta: float | int = 1,
    ) -> None:
        super().__init__(parent=parent, label=label, units=units)
        self._start: float | int = start
        self._delta: float | int = delta

    def __repr__(self) -> str:
        return (
            "NMScaleX(label='%s', units='%s', start=%s, delta=%s)"
            % (self._label, self._units, self._start, self._delta)
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NMScaleX):
            return (
                self._label == other._label
                and self._units == other._units
                and self._start == other._start
                and self._delta == other._delta
            )
        if isinstance(other, dict):
            return self.to_dict() == other
        return NotImplemented

    # --- start ---

    @property
    def start(self) -> float | int:
        return self._start

    @start.setter
    def start(self, value: float | int) -> None:
        self._set_start(value)

    def _set_start(
        self,
        value: float | int,
        log: bool = True,
        quiet: bool = nmp.QUIET,
    ) -> None:
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "start", "number"))
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            raise ValueError("start: %s" % value)
        if value == self._start:
            return
        self._start = value
        if log:
            nmh.history(
                "set start=%s" % value,
                path=self.path_str,
                quiet=quiet,
            )

    # --- delta ---

    @property
    def delta(self) -> float | int:
        return self._delta

    @delta.setter
    def delta(self, value: float | int) -> None:
        self._set_delta(value)

    def _set_delta(
        self,
        value: float | int,
        log: bool = True,
        quiet: bool = nmp.QUIET,
    ) -> None:
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "delta", "number"))
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            raise ValueError("delta: %s" % value)
        if value == 0:
            raise ValueError("delta cannot be zero")
        if value == self._delta:
            return
        self._delta = value
        if log:
            nmh.history(
                "set delta=%s" % value,
                path=self.path_str,
                quiet=quiet,
            )

    # --- serialization ---

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["start"] = self._start
        d["delta"] = self._delta
        return d


# --- Helper functions for constructors ---


def _xscale_from_dict(
    d: dict | None,
    parent: object | None = None,
) -> NMScaleX:
    if d is None:
        return NMScaleX(parent=parent)
    if not isinstance(d, dict):
        raise TypeError(nmu.type_error_str(d, "xscale", "dict"))
    return NMScaleX(
        parent=parent,
        label=d.get("label", ""),
        units=d.get("units", ""),
        start=d.get("start", 0),
        delta=d.get("delta", 1),
    )


def _yscale_from_dict(
    d: dict | None,
    parent: object | None = None,
) -> NMScaleY:
    if d is None:
        return NMScaleY(parent=parent)
    if not isinstance(d, dict):
        raise TypeError(nmu.type_error_str(d, "yscale", "dict"))
    return NMScaleY(
        parent=parent,
        label=d.get("label", ""),
        units=d.get("units", ""),
    )
