# -*- coding: utf-8 -*-
"""
NMPulse and NMPulseContainer: single-pulse component and ordered container.

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
import numpy.random as _rng

import pyneuromatic.core.nm_configurations as nmc
import pyneuromatic.core.nm_history as nmh
from pyneuromatic.core.nm_command_history import add_nm_command
import pyneuromatic.core.nm_utilities as nmu
from pyneuromatic.analysis.nm_pulse_func import (
    NMPulseFunc,
    NMPulseFuncSquare,
    _VALID_SHAPES,
    _pulse_func_from_dict,
)

_FLOAT_ATTRS: tuple[str, ...] = (
    "amp", "onset", "width",
    "amp_delta", "onset_delta", "width_delta",
    "amp_stdv", "onset_stdv", "width_stdv",
)

# Keys that belong to the NMPulseFunc subclass, not to NMPulse directly.
_FUNC_ONLY_KEYS: frozenset[str] = frozenset({
    "tau", "freq", "phase", "f0", "f1", "data", "xdelta_data",
})


class NMPulse:
    """Single pulse component: shape, parameters, and epoch targeting.

    Lightweight class (does not inherit NMObject) following the NMStatWin
    pattern. Each NMPulse defines one waveform component that can be added
    to an output epoch array.

    Epoch targeting:
        - ``epoch="all"`` (default): fires on every epoch (stride controlled
          by ``epoch_delta``).
        - ``epoch=N``: fires starting at epoch N, then every ``epoch_delta``
          epochs.

    Parameter variation per occurrence (0-based firing count):
        - ``delta_<param>``: linear shift per occurrence.
        - ``stdv_<param>``: Gaussian noise σ added each occurrence.

    Args:
        name: Pulse name (default ``"NMPulse0"``).
        config: Optional dict of initial parameter values.
        nm_path: Dot-path used in command history entries.
    """

    def __init__(
        self,
        name: str = "NMPulse0",
        config: dict | None = None,
        nm_path: str = "pulse.pulses",
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        self._name = name
        self._nm_path = nm_path

        self._func: NMPulseFunc = NMPulseFuncSquare()
        self._epoch: int = 0
        self._epoch_delta: int = 0
        self._amp: float = 1.0
        self._onset: float = 10.0
        self._width: float = math.inf
        self._amp_delta: float = 0.0
        self._onset_delta: float = 0.0
        self._width_delta: float = 0.0
        self._amp_stdv: float = 0.0
        self._onset_stdv: float = 0.0
        self._width_stdv: float = 0.0

        if config is not None:
            self._config_set(config, quiet=True)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMPulse):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    @property
    def name(self) -> str:
        """Pulse name string."""
        return self._name

    # ------------------------------------------------------------------
    # pulse shape / func

    @property
    def func(self) -> NMPulseFunc:
        """The NMPulseFunc instance holding shape-specific parameters."""
        return self._func

    @property
    def pulse(self) -> str:
        """Pulse shape name (e.g. ``"square"``, ``"sin"``, ``"user"``)."""
        return self._func.name

    @pulse.setter
    def pulse(self, value: str) -> None:
        self._pulse_set(value)
        add_nm_command("%s[%r].pulse = %r" % (self._nm_path, self._name, value))

    def _pulse_set(self, value: str, quiet: bool = nmc.QUIET) -> None:
        if not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "pulse", "string"))
        if value not in _VALID_SHAPES:
            raise ValueError(
                "pulse must be one of %s, got %r" % (sorted(_VALID_SHAPES), value)
            )
        self._func = _pulse_func_from_dict({"pulse": value})
        nmh.history("set pulse=%r" % value, path=self._name, quiet=quiet)

    # ------------------------------------------------------------------
    # epoch targeting

    @property
    def epoch(self) -> int:
        """Starting epoch index (0-based). Use ``"all"`` as shorthand for ``epoch=0, epoch_delta=1``."""
        return self._epoch

    @epoch.setter
    def epoch(self, value: int | str) -> None:
        self._epoch_set(value)
        add_nm_command("%s[%r].epoch = %r" % (self._nm_path, self._name, self._epoch))

    def _epoch_set(self, value: int | str, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "epoch", "int or 'all'"))
        if isinstance(value, str):
            if value != "all":
                raise ValueError("epoch string must be 'all', got %r" % value)
            self._epoch = 0
            self._epoch_delta = 1
            nmh.history("set epoch='all' (epoch=0, epoch_delta=1)", path=self._name, quiet=quiet)
            return
        if not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "epoch", "int or 'all'"))
        if value < 0:
            raise ValueError("epoch must be >= 0, got %d" % value)
        self._epoch = value
        nmh.history("set epoch=%d" % self._epoch, path=self._name, quiet=quiet)

    @property
    def epoch_delta(self) -> int:
        """Stride between firings (>= 0). ``0`` fires only once; ``2`` fires every other epoch."""
        return self._epoch_delta

    @epoch_delta.setter
    def epoch_delta(self, value: int) -> None:
        self._epoch_delta_set(value)
        add_nm_command(
            "%s[%r].epoch_delta = %r" % (self._nm_path, self._name, self._epoch_delta)
        )

    def _epoch_delta_set(self, value: int, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "epoch_delta", "int"))
        if value < 0:
            raise ValueError("epoch_delta must be >= 0, got %d" % value)
        self._epoch_delta = value
        nmh.history("set epoch_delta=%d" % self._epoch_delta, path=self._name, quiet=quiet)

    # ------------------------------------------------------------------
    # float parameters (amp, onset, width, delta_*, stdv_*)

    def _float_set(self, attr: str, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, attr, "float"))
        setattr(self, "_" + attr, float(value))
        nmh.history("set %s=%g" % (attr, float(value)), path=self._name, quiet=quiet)

    @property
    def amp(self) -> float:
        return self._amp

    @amp.setter
    def amp(self, value: float) -> None:
        self._float_set("amp", value)
        add_nm_command("%s[%r].amp = %r" % (self._nm_path, self._name, self._amp))

    @property
    def onset(self) -> float:
        return self._onset

    @onset.setter
    def onset(self, value: float) -> None:
        self._float_set("onset", value)
        add_nm_command("%s[%r].onset = %r" % (self._nm_path, self._name, self._onset))

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        self._float_set("width", value)
        add_nm_command("%s[%r].width = %r" % (self._nm_path, self._name, self._width))

    @property
    def amp_delta(self) -> float:
        return self._amp_delta

    @amp_delta.setter
    def amp_delta(self, value: float) -> None:
        self._float_set("amp_delta", value)
        add_nm_command(
            "%s[%r].amp_delta = %r" % (self._nm_path, self._name, self._amp_delta)
        )

    @property
    def onset_delta(self) -> float:
        return self._onset_delta

    @onset_delta.setter
    def onset_delta(self, value: float) -> None:
        self._float_set("onset_delta", value)
        add_nm_command(
            "%s[%r].onset_delta = %r" % (self._nm_path, self._name, self._onset_delta)
        )

    @property
    def width_delta(self) -> float:
        return self._width_delta

    @width_delta.setter
    def width_delta(self, value: float) -> None:
        self._float_set("width_delta", value)
        add_nm_command(
            "%s[%r].width_delta = %r" % (self._nm_path, self._name, self._width_delta)
        )

    @property
    def amp_stdv(self) -> float:
        return self._amp_stdv

    @amp_stdv.setter
    def amp_stdv(self, value: float) -> None:
        self._float_set("amp_stdv", value)
        add_nm_command(
            "%s[%r].amp_stdv = %r" % (self._nm_path, self._name, self._amp_stdv)
        )

    @property
    def onset_stdv(self) -> float:
        return self._onset_stdv

    @onset_stdv.setter
    def onset_stdv(self, value: float) -> None:
        self._float_set("onset_stdv", value)
        add_nm_command(
            "%s[%r].onset_stdv = %r" % (self._nm_path, self._name, self._onset_stdv)
        )

    @property
    def width_stdv(self) -> float:
        return self._width_stdv

    @width_stdv.setter
    def width_stdv(self, value: float) -> None:
        self._float_set("width_stdv", value)
        add_nm_command(
            "%s[%r].width_stdv = %r" % (self._nm_path, self._name, self._width_stdv)
        )

    # ------------------------------------------------------------------
    # Epoch targeting

    def targets_epoch(self, epoch_idx: int) -> bool:
        """Return True if this pulse fires on epoch *epoch_idx*.

        When ``epoch_delta=0``: fires only on ``epoch``.
        When ``epoch_delta=M``: fires on ``epoch``, ``epoch+M``, ``epoch+2M``, ...
        """
        if epoch_idx < self._epoch:
            return False
        if self._epoch_delta == 0:
            return epoch_idx == self._epoch
        return (epoch_idx - self._epoch) % self._epoch_delta == 0

    def _occurrence_idx(self, epoch_idx: int) -> int:
        """Return the 0-based firing count for *epoch_idx* (used to scale delta_*)."""
        if self._epoch_delta == 0:
            return 0
        return (epoch_idx - self._epoch) // self._epoch_delta

    # ------------------------------------------------------------------
    # Waveform generation

    def waveform(
        self,
        n_points: int,
        xstart: float,
        xdelta: float,
        epoch_idx: int,
    ) -> np.ndarray:
        """Compute this pulse's contribution for *epoch_idx*.

        Scales delta_* by occurrence index and adds stdv noise, then
        delegates to the appropriate nm_math.pulse_* function.

        Args:
            n_points: Number of samples.
            xstart: X-value of the first sample.
            xdelta: Sample interval.
            epoch_idx: 0-based epoch counter from NMToolPulse.run().

        Returns:
            1-D float64 array of length *n_points*.
        """
        occ = self._occurrence_idx(epoch_idx)
        amp   = self._amp   + self._amp_delta   * occ + (
            _rng.normal(0.0, self._amp_stdv)   if self._amp_stdv   > 0 else 0.0)
        onset = self._onset + self._onset_delta * occ + (
            _rng.normal(0.0, self._onset_stdv) if self._onset_stdv > 0 else 0.0)
        width = self._width + self._width_delta * occ + (
            _rng.normal(0.0, self._width_stdv) if self._width_stdv > 0 else 0.0)

        return self._func.waveform(n_points, xstart, xdelta, amp, onset, width)

    # ------------------------------------------------------------------
    # Serialisation

    def _config_set(self, config: dict, quiet: bool = nmc.QUIET) -> None:
        """Set multiple parameters from a dict. Unknown keys raise KeyError.

        Two-phase: shape name + func-only keys (tau, freq, phase, f0, f1,
        data, xdelta_data) are used to create/update the NMPulseFunc; all
        other keys (amp, onset, width, delta_*, stdv_*, epoch, epoch_delta)
        are applied to NMPulse attrs directly.
        """
        if not isinstance(config, dict):
            raise TypeError(nmu.type_error_str(config, "config", "dict"))

        # Validate all keys up front
        pulse_name = None
        func_kwargs: dict = {}
        nm_attrs: dict = {}
        for k, v in config.items():
            if not isinstance(k, str):
                raise TypeError(nmu.type_error_str(k, "key", "string"))
            kl = k.lower()
            if kl == "name":
                continue
            elif kl == "pulse":
                pulse_name = v
            elif kl in _FUNC_ONLY_KEYS:
                func_kwargs[kl] = v
            elif kl in ("epoch", "epoch_delta") or kl in _FLOAT_ATTRS:
                nm_attrs[kl] = v
            else:
                raise KeyError("NMPulse: unknown config key %r" % k)

        # Phase 1: update NMPulseFunc if pulse name or func-only keys changed
        if pulse_name is not None or func_kwargs:
            name = pulse_name if pulse_name is not None else self._func.name
            self._func = _pulse_func_from_dict({"pulse": name, **func_kwargs})

        # Phase 2: apply NMPulse scalar attrs
        for kl, v in nm_attrs.items():
            if kl == "epoch":
                self._epoch_set(v, quiet=True)
            elif kl == "epoch_delta":
                self._epoch_delta_set(v, quiet=True)
            else:
                self._float_set(kl, v, quiet=True)

        nmh.history("set config=%s" % config, path=self._name, quiet=quiet)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON/TOML-safe).

        The ``"pulse"`` key and any shape-specific keys (e.g. ``"freq"``,
        ``"tau"``) come from the embedded :class:`NMPulseFunc`.  Universal
        parameters (amp, onset, width, tau) and variation parameters are
        stored directly on NMPulse and take precedence over any overlapping
        values in the func's dict.
        """
        d: dict = {
            "name":          self._name,
            "epoch":         self._epoch,
            "epoch_delta":   self._epoch_delta,
            "amp":           self._amp,
            "onset":         self._onset,
            "width":         self._width,
            "amp_delta":     self._amp_delta,
            "onset_delta":   self._onset_delta,
            "width_delta":   self._width_delta,
            "amp_stdv":      self._amp_stdv,
            "onset_stdv":    self._onset_stdv,
            "width_stdv":    self._width_stdv,
        }
        # Merge func-specific entries (pulse name + shape params like tau/freq/phase).
        d.update(self._func.to_dict())
        return d

    @classmethod
    def from_dict(cls, d: dict, nm_path: str = "pulse.pulses") -> "NMPulse":
        """Reconstruct an NMPulse from a dict produced by ``to_dict()``."""
        name = d.get("name", "NMPulse0")
        p = cls(name=name, nm_path=nm_path)
        p._config_set({k: v for k, v in d.items() if k != "name"}, quiet=True)
        return p


class NMPulseContainer:
    """Ordered container of NMPulse objects with auto-naming.

    Pulses are created via ``new()`` and named ``"p0"``, ``"p1"``, etc.

    Args:
        nm_path: Dot-path used in command history entries.
    """

    def __init__(self, nm_path: str = "pulse.pulses") -> None:
        self._nm_path = nm_path
        self._pulses: dict[str, NMPulse] = {}
        self._count: int = 0

    def new(
        self,
        config: dict | None = None,
        quiet: bool = nmc.QUIET,
    ) -> NMPulse:
        """Create, register, and return a new NMPulse with an auto-name.

        Args:
            config: Optional dict of initial parameter values (passed to
                ``NMPulse._config_set()``).

        Returns:
            The new NMPulse.
        """
        name = "p%d" % self._count
        self._count += 1
        p = NMPulse(name=name, config=config, nm_path=self._nm_path)
        self._pulses[name] = p
        nmh.history("new NMPulse=%s" % name, quiet=quiet)
        add_nm_command("%s.new(%r)" % (self._nm_path, name))
        return p

    def __iter__(self):
        """Iterate over NMPulse objects in insertion order."""
        return iter(self._pulses.values())

    def __len__(self) -> int:
        return len(self._pulses)

    def __getitem__(self, name: str) -> NMPulse:
        return self._pulses[name]

    def __contains__(self, name: str) -> bool:
        return name in self._pulses

    def to_dict(self) -> dict:
        """Serialise the container to a plain dict."""
        return {"pulses": [p.to_dict() for p in self._pulses.values()]}

    @classmethod
    def from_dict(cls, d: dict, nm_path: str = "pulse.pulses") -> "NMPulseContainer":
        """Reconstruct a container from a dict produced by ``to_dict()``."""
        container = cls(nm_path=nm_path)
        for pd in d.get("pulses", []):
            name = pd.get("name", "p%d" % container._count)
            p = NMPulse.from_dict(pd, nm_path=nm_path)
            p._name = name
            container._pulses[name] = p
            container._count += 1
        return container
