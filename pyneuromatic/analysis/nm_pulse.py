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
    "amp", "onset", "duration",
    "amp_delta", "onset_delta", "duration_delta",
    "amp_stdv", "onset_stdv", "duration_stdv",
    "interval", "interval_stdv", "interval_min", "interval_max", "train_duration",
    "rp_taur", "rp_rinf", "rp_rmin", "rp_taup", "rp_pinf", "rp_pmax", "rp_pscale",
    "df_taud", "df_dinf", "df_dmin", "df_dscale", "df_tauf", "df_finf", "df_fmax", "df_fscale",
)

# Keys that belong to the NMPulseFunc subclass, not to NMPulse directly.
_FUNC_ONLY_KEYS: frozenset[str] = frozenset({
    "tau", "freq", "phase", "f0", "f1", "data", "data_xdelta",
})

_VALID_INTERVAL_TYPES: frozenset[str] = frozenset({"fixed", "gaussian", "poisson"})
_VALID_AMP_DISTS:     frozenset[str] = frozenset({"gaussian", "gamma"})


def gamma_params_from_moments(mean: float, stdv: float) -> tuple[float, float]:
    """Return (shape k, scale θ) for a Gamma with the given mean and stdv.

    Inverse of :func:`gamma_moments_from_params`. Use to convert ``amp`` and
    ``amp_stdv`` values into Gamma distribution parameters.
    """
    k = (mean / stdv) ** 2
    theta = stdv ** 2 / mean
    return k, theta


def gamma_moments_from_params(k: float, theta: float) -> tuple[float, float]:
    """Return (mean, stdv) for a Gamma with shape k and scale θ.

    Inverse of :func:`gamma_params_from_moments`. Use to find the ``amp`` and
    ``amp_stdv`` values that correspond to known Gamma parameters.
    """
    return k * theta, math.sqrt(k) * theta


class NMPulse:
    """Pulse component: shape, parameters, epoch targeting, and optional train.

    Lightweight class (does not inherit NMObject) following the NMStatWin
    pattern. Each NMPulse defines one waveform component that can be added
    to an output epoch array.

    When ``n_pulses=1`` (default) a single pulse is generated.  When
    ``n_pulses>1`` a train of that pulse shape is generated with inter-pulse
    intervals controlled by ``interval`` and ``interval_type``.

    Epoch targeting:
        - ``epoch=0, epoch_delta=0`` (default): fires only on epoch 0.
        - ``epoch="all"``: shorthand for ``epoch=0, epoch_delta=1``.

    Train parameters (ignored when ``n_pulses=1``):
        - ``interval``: mean inter-pulse interval (> 0). Default 100.0.
        - ``interval_type``: ``"fixed"`` (default), ``"gaussian"``, or
          ``"poisson"``.
        - ``interval_stdv``: stdv for Gaussian jitter (``interval_type="gaussian"``
          only). Default 0.0.

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
        self._enabled: bool = True
        self._epoch: int = 0
        self._epoch_delta: int = 0
        self._amp: float = 1.0
        self._amp_delta: float = 0.0
        self._amp_stdv: float = 0.0
        self._amp_dist: str = "gaussian"
        self._onset: float = 10.0
        self._onset_delta: float = 0.0
        self._onset_stdv: float = 0.0
        self._duration: float = math.inf
        self._duration_delta: float = 0.0
        self._duration_stdv: float = 0.0
        self._n_pulses: int = 1
        self._interval: float = 100.0
        self._interval_stdv: float = 0.0
        self._interval_min: float = 0.0
        self._interval_max: float = math.inf
        self._interval_type: str = "fixed"
        self._seed: int | None = None
        self._train_duration: float = math.inf
        self._binomial_n: int = 0
        self._binomial_p: float = 1.0
        # R*P short-term plasticity model.
        # Depression only (rp_pscale=0): Tsodyks & Markram 1997.
        # Depression + facilitation (rp_pscale>0, rp_taup>0): Tsodyks, Pawelzik & Markram 1998.
        # rp_taur > 0 enables the model.
        self._rp_taur:   float = 0.0     # recovery tau for R (>0 enables model)
        self._rp_rinf:   float = 1.0     # steady-state R
        self._rp_rmin:   float = 0.0     # minimum R after depletion
        self._rp_pinf:   float = 0.5     # steady-state P (release probability)
        self._rp_pmax:   float = math.inf  # ceiling on P
        self._rp_taup:   float = 0.0     # recovery tau for P (>0 enables P dynamics)
        self._rp_pscale: float = 0.0     # post-pulse P increment scale (>0 enables facilitation)
        # D*F short-term plasticity model (multiplicative depression and facilitation).
        # df_taud > 0 enables the model.
        self._df_taud:  float = 0.0      # recovery tau for D (>0 enables model)
        self._df_dinf:  float = 1.0      # steady-state D
        self._df_dmin:  float = 0.0      # minimum D after depletion
        self._df_dscale: float = 1.0     # post-pulse D multiplier (<1 causes depression)
        self._df_tauf:  float = 0.0      # recovery tau for F (>0 enables F dynamics)
        self._df_finf:  float = 1.0      # steady-state F
        self._df_fmax:  float = math.inf # ceiling on F
        self._df_fscale: float = 1.0     # post-pulse F multiplier (>1 causes facilitation)

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

    @property
    def enabled(self) -> bool:
        """Whether this pulse contributes to the output waveform. Default True."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(nmu.type_error_str(value, "enabled", "bool"))
        self._enabled = value
        nmh.history("set enabled=%s" % value, path=self._name)
        add_nm_command("%s[%r].enabled = %r" % (self._nm_path, self._name, self._enabled))

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
    # float parameters (amp, onset, duration, *_delta, *_stdv, interval*)

    def _float_set(self, attr: str, value: float, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, attr, "float"))
        v = float(value)
        if attr == "interval" and v <= 0:
            raise ValueError("interval must be > 0, got %s" % v)
        if attr == "interval_stdv" and v < 0:
            raise ValueError("interval_stdv must be >= 0, got %s" % v)
        if attr == "interval_min" and v < 0:
            raise ValueError("interval_min must be >= 0, got %s" % v)
        if attr == "interval_max" and v <= 0:
            raise ValueError("interval_max must be > 0, got %s" % v)
        if attr == "train_duration" and v <= 0:
            raise ValueError("train_duration must be > 0, got %s" % v)
        setattr(self, "_" + attr, v)
        nmh.history("set %s=%g" % (attr, v), path=self._name, quiet=quiet)

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
    def duration(self) -> float:
        return self._duration

    @duration.setter
    def duration(self, value: float) -> None:
        self._float_set("duration", value)
        add_nm_command("%s[%r].duration = %r" % (self._nm_path, self._name, self._duration))

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
    def duration_delta(self) -> float:
        return self._duration_delta

    @duration_delta.setter
    def duration_delta(self, value: float) -> None:
        self._float_set("duration_delta", value)
        add_nm_command(
            "%s[%r].duration_delta = %r" % (self._nm_path, self._name, self._duration_delta)
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
    def duration_stdv(self) -> float:
        return self._duration_stdv

    @duration_stdv.setter
    def duration_stdv(self, value: float) -> None:
        self._float_set("duration_stdv", value)
        add_nm_command(
            "%s[%r].duration_stdv = %r" % (self._nm_path, self._name, self._duration_stdv)
        )

    @property
    def n_pulses(self) -> int:
        """Number of pulses in train. ``0`` means unlimited (use ``train_duration`` to bound)."""
        return self._n_pulses

    @n_pulses.setter
    def n_pulses(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "n_pulses", "int"))
        if value < 0:
            raise ValueError("n_pulses must be >= 0, got %d" % value)
        self._n_pulses = value
        add_nm_command("%s[%r].n_pulses = %r" % (self._nm_path, self._name, self._n_pulses))

    @property
    def interval(self) -> float:
        """Mean inter-pulse interval (> 0). Used when n_pulses > 1."""
        return self._interval

    @interval.setter
    def interval(self, value: float) -> None:
        self._float_set("interval", value)
        add_nm_command("%s[%r].interval = %r" % (self._nm_path, self._name, self._interval))

    @property
    def interval_stdv(self) -> float:
        """Gaussian stdv on interval (interval_type='gaussian' only)."""
        return self._interval_stdv

    @interval_stdv.setter
    def interval_stdv(self, value: float) -> None:
        self._float_set("interval_stdv", value)
        add_nm_command(
            "%s[%r].interval_stdv = %r" % (self._nm_path, self._name, self._interval_stdv)
        )

    @property
    def interval_min(self) -> float:
        """Minimum interval after sampling (>= 0). Clamps random draws. Default 0.0."""
        return self._interval_min

    @interval_min.setter
    def interval_min(self, value: float) -> None:
        self._float_set("interval_min", value)
        add_nm_command(
            "%s[%r].interval_min = %r" % (self._nm_path, self._name, self._interval_min)
        )

    @property
    def interval_max(self) -> float:
        """Maximum interval after sampling (> 0). Clamps random draws. Default inf."""
        return self._interval_max

    @interval_max.setter
    def interval_max(self, value: float) -> None:
        self._float_set("interval_max", value)
        add_nm_command(
            "%s[%r].interval_max = %r" % (self._nm_path, self._name, self._interval_max)
        )

    @property
    def interval_type(self) -> str:
        """Interval distribution: ``'fixed'``, ``'gaussian'``, or ``'poisson'``."""
        return self._interval_type

    @interval_type.setter
    def interval_type(self, value: str) -> None:
        self._interval_type_set(value)
        add_nm_command(
            "%s[%r].interval_type = %r" % (self._nm_path, self._name, self._interval_type)
        )

    @property
    def seed(self) -> int | None:
        """Optional RNG seed for stochastic interval types. ``None`` uses the global state."""
        return self._seed

    @seed.setter
    def seed(self, value: int | None) -> None:
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(nmu.type_error_str(value, "seed", "int or None"))
            if value < 0:
                raise ValueError("seed must be >= 0, got %d" % value)
        self._seed = value
        nmh.history("set seed=%r" % value, path=self._name)
        add_nm_command("%s[%r].seed = %r" % (self._nm_path, self._name, self._seed))

    @property
    def train_duration(self) -> float:
        """Duration of the train window. Pulses stop when onset_i >= onset + train_duration.
        Default ``inf`` (no window limit). Requires ``n_pulses=0`` or a large count."""
        return self._train_duration

    @train_duration.setter
    def train_duration(self, value: float) -> None:
        self._float_set("train_duration", value)
        add_nm_command(
            "%s[%r].train_duration = %r" % (self._nm_path, self._name, self._train_duration)
        )

    @property
    def binomial_n(self) -> int:
        """Number of release sites for binomial quantal release. 0 = disabled."""
        return self._binomial_n

    @binomial_n.setter
    def binomial_n(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(nmu.type_error_str(value, "binomial_n", "int"))
        if value < 0:
            raise ValueError("binomial_n must be >= 0, got %d" % value)
        self._binomial_n = value
        nmh.history("set binomial_n=%d" % value, path=self._name)
        add_nm_command("%s[%r].binomial_n = %d" % (self._nm_path, self._name, self._binomial_n))

    @property
    def binomial_p(self) -> float:
        """Probability of release per site (0 < p <= 1). Used when binomial_n > 0."""
        return self._binomial_p

    @binomial_p.setter
    def binomial_p(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(nmu.type_error_str(value, "binomial_p", "float"))
        value = float(value)
        if value <= 0 or value > 1:
            raise ValueError("binomial_p must be in (0, 1], got %g" % value)
        self._binomial_p = value
        nmh.history("set binomial_p=%g" % value, path=self._name)
        add_nm_command("%s[%r].binomial_p = %g" % (self._nm_path, self._name, self._binomial_p))

    # ------------------------------------------------------------------
    # R*P short-term plasticity model

    @property
    def rp_taur(self) -> float:
        """Recovery time constant for R. >0 enables the R*P model; 0 = disabled."""
        return self._rp_taur

    @rp_taur.setter
    def rp_taur(self, value: float) -> None:
        self._float_set("rp_taur", value)
        add_nm_command("%s[%r].rp_taur = %g" % (self._nm_path, self._name, self._rp_taur))

    @property
    def rp_rinf(self) -> float:
        """Steady-state value of R (vesicle pool fraction). Default 1.0."""
        return self._rp_rinf

    @rp_rinf.setter
    def rp_rinf(self, value: float) -> None:
        self._float_set("rp_rinf", value)
        add_nm_command("%s[%r].rp_rinf = %g" % (self._nm_path, self._name, self._rp_rinf))

    @property
    def rp_rmin(self) -> float:
        """Minimum R after depletion. Default 0.0."""
        return self._rp_rmin

    @rp_rmin.setter
    def rp_rmin(self, value: float) -> None:
        self._float_set("rp_rmin", value)
        add_nm_command("%s[%r].rp_rmin = %g" % (self._nm_path, self._name, self._rp_rmin))

    @property
    def rp_pinf(self) -> float:
        """Steady-state release probability P. Default 0.5."""
        return self._rp_pinf

    @rp_pinf.setter
    def rp_pinf(self, value: float) -> None:
        self._float_set("rp_pinf", value)
        add_nm_command("%s[%r].rp_pinf = %g" % (self._nm_path, self._name, self._rp_pinf))

    @property
    def rp_pmax(self) -> float:
        """Ceiling on P (prevents P exceeding this after facilitation). Default inf."""
        return self._rp_pmax

    @rp_pmax.setter
    def rp_pmax(self, value: float) -> None:
        self._float_set("rp_pmax", value)
        add_nm_command("%s[%r].rp_pmax = %g" % (self._nm_path, self._name, self._rp_pmax))

    @property
    def rp_taup(self) -> float:
        """Recovery time constant for P. >0 enables P dynamics; 0 = P fixed at rp_pinf."""
        return self._rp_taup

    @rp_taup.setter
    def rp_taup(self, value: float) -> None:
        self._float_set("rp_taup", value)
        add_nm_command("%s[%r].rp_taup = %g" % (self._nm_path, self._name, self._rp_taup))

    @property
    def rp_pscale(self) -> float:
        """Post-pulse P facilitation scale. >0 enables facilitation via P += rp_pscale*(1-P)."""
        return self._rp_pscale

    @rp_pscale.setter
    def rp_pscale(self, value: float) -> None:
        self._float_set("rp_pscale", value)
        add_nm_command("%s[%r].rp_pscale = %g" % (self._nm_path, self._name, self._rp_pscale))

    @property
    def df_taud(self) -> float:
        """Recovery time constant for D. >0 enables the D*F model; 0 = disabled."""
        return self._df_taud

    @df_taud.setter
    def df_taud(self, value: float) -> None:
        self._float_set("df_taud", value)
        add_nm_command("%s[%r].df_taud = %g" % (self._nm_path, self._name, self._df_taud))

    @property
    def df_dinf(self) -> float:
        """Steady-state value of D. Default 1.0."""
        return self._df_dinf

    @df_dinf.setter
    def df_dinf(self, value: float) -> None:
        self._float_set("df_dinf", value)
        add_nm_command("%s[%r].df_dinf = %g" % (self._nm_path, self._name, self._df_dinf))

    @property
    def df_dmin(self) -> float:
        """Minimum D after depletion. Default 0.0."""
        return self._df_dmin

    @df_dmin.setter
    def df_dmin(self, value: float) -> None:
        self._float_set("df_dmin", value)
        add_nm_command("%s[%r].df_dmin = %g" % (self._nm_path, self._name, self._df_dmin))

    @property
    def df_dscale(self) -> float:
        """Post-pulse D multiplier. <1 causes depression; 1 = no change. Default 1.0."""
        return self._df_dscale

    @df_dscale.setter
    def df_dscale(self, value: float) -> None:
        self._float_set("df_dscale", value)
        add_nm_command("%s[%r].df_dscale = %g" % (self._nm_path, self._name, self._df_dscale))

    @property
    def df_tauf(self) -> float:
        """Recovery time constant for F. >0 enables F dynamics; 0 = F fixed at df_finf."""
        return self._df_tauf

    @df_tauf.setter
    def df_tauf(self, value: float) -> None:
        self._float_set("df_tauf", value)
        add_nm_command("%s[%r].df_tauf = %g" % (self._nm_path, self._name, self._df_tauf))

    @property
    def df_finf(self) -> float:
        """Steady-state value of F. Default 1.0."""
        return self._df_finf

    @df_finf.setter
    def df_finf(self, value: float) -> None:
        self._float_set("df_finf", value)
        add_nm_command("%s[%r].df_finf = %g" % (self._nm_path, self._name, self._df_finf))

    @property
    def df_fmax(self) -> float:
        """Ceiling on F (prevents F exceeding this after facilitation). Default inf."""
        return self._df_fmax

    @df_fmax.setter
    def df_fmax(self, value: float) -> None:
        self._float_set("df_fmax", value)
        add_nm_command("%s[%r].df_fmax = %g" % (self._nm_path, self._name, self._df_fmax))

    @property
    def df_fscale(self) -> float:
        """Post-pulse F multiplier. >1 causes facilitation; 1 = no change. Default 1.0."""
        return self._df_fscale

    @df_fscale.setter
    def df_fscale(self, value: float) -> None:
        self._float_set("df_fscale", value)
        add_nm_command("%s[%r].df_fscale = %g" % (self._nm_path, self._name, self._df_fscale))

    @property
    def amp_dist(self) -> str:
        """Amplitude noise distribution: ``"gaussian"`` (default) or ``"gamma"``.

        When ``amp_stdv > 0`` and ``amp_dist="gamma"``, amplitude is drawn from
        a Gamma distribution with the same mean (``amp``) and stdv (``amp_stdv``)
        as the Gaussian case — always positive, right-skewed.  Use
        :func:`gamma_params_from_moments` to inspect the implied k and θ.

        Requires ``amp > 0`` at waveform generation time.
        """
        return self._amp_dist

    @amp_dist.setter
    def amp_dist(self, value: str) -> None:
        self._amp_dist_set(value)
        add_nm_command("%s[%r].amp_dist = %r" % (self._nm_path, self._name, self._amp_dist))

    def _amp_dist_set(self, value: str, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "amp_dist", "string"))
        if value not in _VALID_AMP_DISTS:
            raise ValueError(
                "amp_dist must be one of %s, got %r" % (sorted(_VALID_AMP_DISTS), value)
            )
        self._amp_dist = value
        nmh.history("set amp_dist=%r" % value, path=self._name, quiet=quiet)

    def _interval_type_set(self, value: str, quiet: bool = nmc.QUIET) -> None:
        if isinstance(value, bool) or not isinstance(value, str):
            raise TypeError(nmu.type_error_str(value, "interval_type", "string"))
        if value not in _VALID_INTERVAL_TYPES:
            raise ValueError(
                "interval_type must be one of %s, got %r"
                % (sorted(_VALID_INTERVAL_TYPES), value)
            )
        self._interval_type = value
        nmh.history("set interval_type=%r" % value, path=self._name, quiet=quiet)

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
        """Return the 0-based firing count for *epoch_idx* (used to scale *_delta)."""
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

        Scales *_delta by occurrence index and adds stdv noise, then
        delegates to the appropriate nm_math.pulse_* function.

        Args:
            n_points: Number of samples.
            xstart: X-value of the first sample.
            xdelta: Sample interval.
            epoch_idx: 0-based epoch counter from NMToolPulse.run().

        Returns:
            1-D float64 array of length *n_points*.
        """
        rng = np.random.default_rng(self._seed) if self._seed is not None else _rng

        occ = self._occurrence_idx(epoch_idx)
        amp_mean = self._amp + self._amp_delta * occ
        if self._amp_stdv > 0:
            if self._amp_dist == "gamma":
                if amp_mean <= 0:
                    raise ValueError(
                        "amp must be > 0 for gamma distribution, got %g" % amp_mean
                    )
                shape, scale = gamma_params_from_moments(amp_mean, self._amp_stdv)
                amp = float(rng.gamma(shape, scale))
            else:
                amp = amp_mean + float(_rng.normal(0.0, self._amp_stdv))
        else:
            amp = amp_mean
        onset = self._onset + self._onset_delta * occ + (
            _rng.normal(0.0, self._onset_stdv) if self._onset_stdv > 0 else 0.0)
        duration = self._duration + self._duration_delta * occ + (
            _rng.normal(0.0, self._duration_stdv) if self._duration_stdv > 0 else 0.0)

        if self._n_pulses == 1:
            self._last_onset_times: list[float] = [onset]
            if self._binomial_n > 0:
                k = int(rng.binomial(self._binomial_n, self._binomial_p))
                self._last_quantal_content: list[int] = [k]
                if k == 0:
                    return np.zeros(n_points, dtype=float)
                return k * self._func.waveform(n_points, xstart, xdelta, amp, onset, duration)
            self._last_quantal_content = [1]
            return self._func.waveform(n_points, xstart, xdelta, amp, onset, duration)

        # train of pulses...

        rp_active = self._rp_taur > 0
        df_active = self._df_taud > 0
        if rp_active and df_active:
            raise ValueError(
                "rp_taur and df_taud cannot both be > 0: "
                "R*P and D*F are alternative plasticity models, not combinable"
            )
        R = self._rp_rinf
        P = self._rp_pinf
        D = self._df_dinf
        F = self._df_finf
        # dummy interval for pulse 0: all state vars start at steady-state so
        # the exp recovery terms evaluate to zero — same result as skipping recovery.
        intvl = self._interval

        y = np.zeros(n_points, dtype=float)
        onset_i = onset
        count = 0
        train_end = onset + self._train_duration
        onset_times: list[float] = []
        quantal_content: list[int] = []
        rp_R: list[float] = []
        rp_P: list[float] = []
        df_D: list[float] = []
        df_F: list[float] = []
        while True:
            if self._n_pulses > 0 and count >= self._n_pulses:
                break
            if not math.isinf(self._train_duration) and onset_i >= train_end:
                break

            effective_amp = amp
            if rp_active:
                R = self._rp_rinf + (R - self._rp_rinf) * math.exp(-intvl / self._rp_taur)
                if self._rp_taup > 0:
                    P = self._rp_pinf + (P - self._rp_pinf) * math.exp(-intvl / self._rp_taup)
                effective_amp *= R * P
            if df_active:
                D = self._df_dinf + (D - self._df_dinf) * math.exp(-intvl / self._df_taud)
                if self._df_tauf > 0:
                    F = self._df_finf + (F - self._df_finf) * math.exp(-intvl / self._df_tauf)
                effective_amp *= D * F

            onset_times.append(onset_i)
            rp_R.append(R)
            rp_P.append(P)
            df_D.append(D)
            df_F.append(F)

            if self._binomial_n > 0:
                k = int(rng.binomial(self._binomial_n, self._binomial_p))
                quantal_content.append(k)
                if k > 0:
                    y += k * self._func.waveform(n_points, xstart, xdelta, effective_amp, onset_i, duration)
            else:
                quantal_content.append(1)
                y += self._func.waveform(n_points, xstart, xdelta, effective_amp, onset_i, duration)
            count += 1

            if rp_active:
                R = max(self._rp_rmin, R * (1.0 - P))
                if self._rp_pscale > 0:
                    P = min(self._rp_pmax, P + self._rp_pscale * (1.0 - P))
            if df_active:
                D = max(self._df_dmin, D * self._df_dscale)
                F = min(self._df_fmax, F * self._df_fscale)

            if self._interval_type == "poisson":
                intvl = rng.exponential(scale=self._interval)
            elif self._interval_type == "gaussian":
                intvl = rng.normal(self._interval, self._interval_stdv)
            else:
                intvl = self._interval
            intvl = max(intvl, self._interval_min)
            intvl = min(intvl, self._interval_max)
            onset_i += intvl

        self._last_onset_times = onset_times
        self._last_quantal_content = quantal_content
        self._last_rp_R: list[float] = rp_R
        self._last_rp_P: list[float] = rp_P
        self._last_df_D: list[float] = df_D
        self._last_df_F: list[float] = df_F
        return y

    # ------------------------------------------------------------------
    # Serialisation

    def _config_set(self, config: dict, quiet: bool = nmc.QUIET) -> None:
        """Set multiple parameters from a dict. Unknown keys raise KeyError.

        Two-phase: shape name + func-only keys (tau, freq, phase, f0, f1,
        data, data_xdelta) are used to create/update the NMPulseFunc; all
        other keys (amp, onset, duration, *_delta, *_stdv, epoch, epoch_delta)
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
            if kl in ("name", "type"):
                continue
            elif kl == "pulse":
                pulse_name = v
            elif kl in _FUNC_ONLY_KEYS:
                func_kwargs[kl] = v
            elif kl in ("epoch", "epoch_delta", "n_pulses", "interval_type", "amp_dist",
                        "enabled", "seed", "binomial_n", "binomial_p") \
                    or kl in _FLOAT_ATTRS:  # includes interval, interval_stdv, train_duration
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
            elif kl == "n_pulses":
                self.n_pulses = v
            elif kl == "interval_type":
                self._interval_type_set(v, quiet=True)
            elif kl == "amp_dist":
                self._amp_dist_set(v, quiet=True)
            elif kl == "enabled":
                self.enabled = v
            elif kl == "seed":
                self.seed = v
            elif kl == "binomial_n":
                self.binomial_n = v
            elif kl == "binomial_p":
                self.binomial_p = v
            else:
                self._float_set(kl, v, quiet=True)

        nmh.history("set config=%s" % config, path=self._name, quiet=quiet)

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON/TOML-safe).

        The ``"pulse"`` key and any shape-specific keys (e.g. ``"freq"``,
        ``"tau"``) come from the embedded :class:`NMPulseFunc`.  Universal
        parameters (amp, onset, duration, tau) and variation parameters are
        stored directly on NMPulse and take precedence over any overlapping
        values in the func's dict.
        """
        d: dict = {
            "name":           self._name,
            "enabled":        self._enabled,
            "seed":           self._seed,
            "epoch":          self._epoch,
            "epoch_delta":    self._epoch_delta,
            "amp":            self._amp,
            "amp_delta":      self._amp_delta,
            "amp_stdv":       self._amp_stdv,
            "amp_dist":       self._amp_dist,
            "onset":          self._onset,
            "onset_delta":    self._onset_delta,
            "onset_stdv":     self._onset_stdv,
            "duration":       self._duration,
            "duration_delta": self._duration_delta,
            "duration_stdv":  self._duration_stdv,
            "n_pulses":       self._n_pulses,
            "interval":       self._interval,
            "interval_stdv":  self._interval_stdv,
            "interval_min":   self._interval_min,
            "interval_max":   self._interval_max,
            "interval_type":  self._interval_type,
            "train_duration":    self._train_duration,
            "binomial_n":        self._binomial_n,
            "binomial_p":        self._binomial_p,
            "rp_taur":           self._rp_taur,
            "rp_rinf":           self._rp_rinf,
            "rp_rmin":           self._rp_rmin,
            "rp_taup":           self._rp_taup,
            "rp_pinf":           self._rp_pinf,
            "rp_pmax":           self._rp_pmax,
            "rp_pscale":         self._rp_pscale,
            "df_taud":           self._df_taud,
            "df_dinf":           self._df_dinf,
            "df_dmin":           self._df_dmin,
            "df_dscale":         self._df_dscale,
            "df_tauf":           self._df_tauf,
            "df_finf":           self._df_finf,
            "df_fmax":           self._df_fmax,
            "df_fscale":         self._df_fscale,
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

    Items are created via ``new()`` and named ``"p0"``, ``"p1"``, etc.

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
            config: Optional dict of initial parameter values.  Set
                ``n_pulses > 1`` for a train.

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
        """Iterate over pulses in insertion order."""
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
