# -*- coding: utf-8 -*-
"""
NMPulseFunc class hierarchy: pulse shape types with parameters and waveform generation.

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

import pyneuromatic.core.nm_utilities as nmu

_VALID_SHAPES: frozenset[str] = frozenset({
    "square", "ramp+", "ramp-", "exp", "alpha",
    "sin", "sinzap", "user",
})


# =========================================================================
# NMPulseFunc base class
# =========================================================================


class NMPulseFunc:
    """Base class for pulse shape types.

    Lightweight class (following NMStatFunc pattern) that represents a pulse
    shape with its shape-specific parameters.  Universal parameters
    (``amp``, ``onset``, ``duration``) are passed as arguments to
    :meth:`waveform` by the caller (:class:`~pyneuromatic.analysis.nm_pulse.NMPulse`)
    so that per-epoch variation (``*_delta``/``*_stdv``) can be applied
    before dispatch.

    Each subclass stores only its own shape-specific parameters and
    implements :meth:`waveform`.

    Args:
        name: Pulse shape name string (e.g. ``"square"``, ``"sin"``).
    """

    _VALID_KEYS: frozenset[str] = frozenset()

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        d = self.to_dict()
        params = ", ".join("%s=%r" % (k, v) for k, v in d.items())
        return "%s(%s)" % (self.__class__.__name__, params)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NMPulseFunc):
            return self.to_dict() == other.to_dict()
        if isinstance(other, dict):
            return self.to_dict() == other
        return NotImplemented

    @property
    def name(self) -> str:
        """Pulse shape name string."""
        return self._name

    def to_dict(self) -> dict:
        """Return shape-specific parameters as a dict with a ``"pulse"`` key."""
        return {"pulse": self._name}

    def _build_masked_x(
        self,
        n_points: int,
        xstart: float,
        xdelta: float,
        onset: float,
        duration: float,
    ) -> tuple:
        """Validate args and build the x-axis, zero output array, and window mask.

        Returns:
            ``(x, y, mask)`` — x-axis array, zero-filled output array, and
            boolean mask selecting samples in ``[onset, onset + duration)``.
            When ``duration`` is ``inf`` the mask selects ``x >= onset``.
        """
        if n_points < 1:
            raise ValueError("n_points must be >= 1, got %d" % n_points)
        if xdelta <= 0:
            raise ValueError("xdelta must be > 0, got %s" % xdelta)
        if duration <= 0:
            raise ValueError("duration must be > 0, got %s" % duration)
        x = xstart + np.arange(n_points) * xdelta
        y = np.zeros(n_points, dtype=float)
        if math.isinf(duration):
            mask = x >= onset
        else:
            mask = (x >= onset) & (x < onset + duration)
        return x, y, mask

    def waveform(
        self,
        n_points: int,
        xstart: float,
        xdelta: float,
        amp: float,
        onset: float,
        duration: float,
        **kwargs,
    ) -> np.ndarray:
        """Generate the waveform for this pulse shape.

        Args:
            n_points: Number of samples.
            xstart: X-value of the first sample.
            xdelta: Sample interval.
            amp: Pulse amplitude (already varied by caller).
            onset: Pulse start x-value (already varied by caller).
            duration: Pulse duration (already varied by caller).
            **kwargs: Shape-specific overrides (e.g. ``tau=`` for exp/alpha).

        Returns:
            1-D float64 array of length *n_points*.
        """
        raise NotImplementedError(
            "%s.waveform() not implemented" % self.__class__.__name__
        )


# =========================================================================
# Subclasses
# =========================================================================


class NMPulseFuncSquare(NMPulseFunc):
    """Square (rectangular) pulse.

    Args:
        name: Must be ``"square"``.
    """

    def __init__(self, name: str = "square") -> None:
        if name != "square":
            raise ValueError("name must be 'square', got %r" % name)
        super().__init__(name)

    def waveform(self, n_points, xstart, xdelta, amp, onset, duration, **_):
        x, y, mask = self._build_masked_x(n_points, xstart, xdelta, onset, duration)
        y[mask] = amp
        return y


class NMPulseFuncRamp(NMPulseFunc):
    """Linear ramp pulse (rising or falling).

    Args:
        name: ``"ramp+"`` for a rising ramp or ``"ramp-"`` for a falling ramp.
    """

    def __init__(self, name: str) -> None:
        if name not in ("ramp+", "ramp-"):
            raise ValueError("name must be 'ramp+' or 'ramp-', got %r" % name)
        super().__init__(name)
        self._direction: str = name[-1]

    def waveform(self, n_points, xstart, xdelta, amp, onset, duration, **_):
        x, y, mask = self._build_masked_x(n_points, xstart, xdelta, onset, duration)
        if not np.any(mask):
            return y
        if math.isinf(duration):
            ew = xstart + (n_points - 1) * xdelta - onset
            if ew <= 0:
                return y
        else:
            ew = duration
        t = (x[mask] - onset) / ew
        y[mask] = amp * t if self._direction == "+" else amp * (1.0 - t)
        return y


class NMPulseFuncExp(NMPulseFunc):
    """Decaying exponential pulse.

    Args:
        name: Must be ``"exp"``.
        tau: Time constant (> 0). Default 10.0.
    """

    _VALID_KEYS: frozenset[str] = frozenset({"tau"})

    def __init__(self, name: str = "exp", tau: float = 10.0) -> None:
        if name != "exp":
            raise ValueError("name must be 'exp', got %r" % name)
        if isinstance(tau, bool) or not isinstance(tau, (int, float)):
            raise TypeError(nmu.type_error_str(tau, "tau", "float"))
        tau = float(tau)
        if tau <= 0:
            raise ValueError("tau must be > 0, got %s" % tau)
        super().__init__(name)
        self._tau: float = tau

    @property
    def tau(self) -> float:
        """Time constant."""
        return self._tau

    def to_dict(self) -> dict:
        return {"pulse": self._name, "tau": self._tau}

    def waveform(self, n_points, xstart, xdelta, amp, onset, duration, **_):
        x, y, mask = self._build_masked_x(n_points, xstart, xdelta, onset, duration)
        y[mask] = amp * np.exp(-(x[mask] - onset) / self._tau)
        return y


class NMPulseFuncAlpha(NMPulseFunc):
    """Alpha-function pulse (peaks at onset + tau).

    Args:
        name: Must be ``"alpha"``.
        tau: Time to peak from onset (> 0). Default 10.0.
    """

    _VALID_KEYS: frozenset[str] = frozenset({"tau"})

    def __init__(self, name: str = "alpha", tau: float = 10.0) -> None:
        if name != "alpha":
            raise ValueError("name must be 'alpha', got %r" % name)
        if isinstance(tau, bool) or not isinstance(tau, (int, float)):
            raise TypeError(nmu.type_error_str(tau, "tau", "float"))
        tau = float(tau)
        if tau <= 0:
            raise ValueError("tau must be > 0, got %s" % tau)
        super().__init__(name)
        self._tau: float = tau

    @property
    def tau(self) -> float:
        """Time to peak."""
        return self._tau

    def to_dict(self) -> dict:
        return {"pulse": self._name, "tau": self._tau}

    def waveform(self, n_points, xstart, xdelta, amp, onset, duration, **_):
        x, y, mask = self._build_masked_x(n_points, xstart, xdelta, onset, duration)
        t = (x[mask] - onset) / self._tau
        y[mask] = amp * t * np.exp(1.0 - t)
        return y


class NMPulseFuncSin(NMPulseFunc):
    """Sinusoidal pulse.

    Args:
        name: Must be ``"sin"``.
        freq: Frequency in cycles per x-unit (> 0). Default 1.0.
        phase: Phase offset in radians. Default 0.0.
    """

    _VALID_KEYS: frozenset[str] = frozenset({"freq", "phase"})

    def __init__(
        self, name: str = "sin", freq: float = 1.0, phase: float = 0.0
    ) -> None:
        if name != "sin":
            raise ValueError("name must be 'sin', got %r" % name)
        freq, phase = _validate_freq_phase(freq, phase)
        super().__init__(name)
        self._freq: float = freq
        self._phase: float = phase

    @property
    def freq(self) -> float:
        """Frequency in cycles per x-unit."""
        return self._freq

    @freq.setter
    def freq(self, value: float) -> None:
        self._freq, _ = _validate_freq_phase(value, self._phase)

    @property
    def phase(self) -> float:
        """Phase offset in radians."""
        return self._phase

    @phase.setter
    def phase(self, value: float) -> None:
        _, self._phase = _validate_freq_phase(self._freq, value)

    def to_dict(self) -> dict:
        return {"pulse": self._name, "freq": self._freq, "phase": self._phase}

    def waveform(self, n_points, xstart, xdelta, amp, onset, duration, **_):
        x, y, mask = self._build_masked_x(n_points, xstart, xdelta, onset, duration)
        y[mask] = amp * np.sin(2.0 * math.pi * self._freq * (x[mask] - onset) + self._phase)
        return y



class NMPulseFuncSinZap(NMPulseFunc):
    """Linear-frequency-sweep (chirp) pulse.

    Frequency sweeps linearly from *f0* to *f1* over the pulse duration.

    Args:
        name: Must be ``"sinzap"``.
        f0: Starting frequency in cycles per x-unit (> 0). Default 1.0.
        f1: Ending frequency in cycles per x-unit (> 0). Default 10.0.
    """

    _VALID_KEYS: frozenset[str] = frozenset({"f0", "f1"})

    def __init__(
        self, name: str = "sinzap", f0: float = 1.0, f1: float = 10.0
    ) -> None:
        if name != "sinzap":
            raise ValueError("name must be 'sinzap', got %r" % name)
        f0, f1 = _validate_f0_f1(f0, f1)
        super().__init__(name)
        self._f0: float = f0
        self._f1: float = f1

    @property
    def f0(self) -> float:
        """Starting frequency."""
        return self._f0

    @f0.setter
    def f0(self, value: float) -> None:
        self._f0, _ = _validate_f0_f1(value, self._f1)

    @property
    def f1(self) -> float:
        """Ending frequency."""
        return self._f1

    @f1.setter
    def f1(self, value: float) -> None:
        _, self._f1 = _validate_f0_f1(self._f0, value)

    def to_dict(self) -> dict:
        return {"pulse": self._name, "f0": self._f0, "f1": self._f1}

    def waveform(self, n_points, xstart, xdelta, amp, onset, duration, **_):
        x, y, mask = self._build_masked_x(n_points, xstart, xdelta, onset, duration)
        if not np.any(mask):
            return y
        if math.isinf(duration):
            ew = xstart + (n_points - 1) * xdelta - onset
            if ew <= 0:
                return y
        else:
            ew = duration
        t = x[mask] - onset
        phi = 2.0 * math.pi * (self._f0 * t + (self._f1 - self._f0) / (2.0 * ew) * t ** 2)
        y[mask] = amp * np.sin(phi)
        return y


class NMPulseFuncUser(NMPulseFunc):
    """User-supplied waveform pulse.

    The stored numpy array is treated as a template: at ``waveform()`` time it
    is normalised so its absolute peak maps to *amp*, shifted to start at
    *onset*, resampled to the target x-scale via ``np.interp``, and truncated
    at ``onset + duration``.

    .. note::
        The data array is **not** included in :meth:`to_dict` output (JSON /
        TOML serialisation limitation).  Save the array separately and
        re-supply it on load.

    Args:
        name: Must be ``"user"``.
        data: 1-D numpy array representing the waveform template (any
            sampling rate).
        data_xdelta: Sample interval of *data* in the same x-units as the
            target waveform. Default 1.0.
    """

    _VALID_KEYS: frozenset[str] = frozenset({"data", "data_xdelta"})

    def __init__(
        self,
        name: str = "user",
        data: np.ndarray | None = None,
        data_xdelta: float = 1.0,
    ) -> None:
        if name != "user":
            raise ValueError("name must be 'user', got %r" % name)
        if data is None:
            raise KeyError("missing key 'data' for NMPulseFuncUser")
        data = np.asarray(data, dtype=float)
        if data.ndim != 1 or len(data) == 0:
            raise ValueError("data must be a non-empty 1-D array")
        if isinstance(data_xdelta, bool) or not isinstance(data_xdelta, (int, float)):
            raise TypeError(nmu.type_error_str(data_xdelta, "data_xdelta", "float"))
        data_xdelta = float(data_xdelta)
        if data_xdelta <= 0:
            raise ValueError("data_xdelta must be > 0, got %s" % data_xdelta)
        super().__init__(name)
        self._data: np.ndarray = data
        self._data_xdelta: float = data_xdelta

    @property
    def data(self) -> np.ndarray:
        """The waveform template array."""
        return self._data

    @property
    def data_xdelta(self) -> float:
        """Sample interval of the template array."""
        return self._data_xdelta

    def to_dict(self) -> dict:
        return {
            "pulse": self._name,
            "data_xdelta": self._data_xdelta,
            "n_data": len(self._data),
        }

    def waveform(self, n_points, xstart, xdelta, amp, onset, duration, **_):
        x, y, mask = self._build_masked_x(n_points, xstart, xdelta, onset, duration)
        if not np.any(mask):
            return y

        peak = np.max(np.abs(self._data))
        if peak == 0.0:
            return y
        data_norm = self._data / peak

        x_src = np.arange(len(self._data)) * self._data_xdelta
        x_tgt = x[mask] - onset
        y_interp = np.interp(x_tgt, x_src, data_norm, left=0.0, right=0.0)
        y[mask] = amp * y_interp
        return y


# =========================================================================
# Validation helpers
# =========================================================================


def _validate_freq_phase(freq: float, phase: float) -> tuple[float, float]:
    if isinstance(freq, bool) or not isinstance(freq, (int, float)):
        raise TypeError(nmu.type_error_str(freq, "freq", "float"))
    freq = float(freq)
    if freq <= 0:
        raise ValueError("freq must be > 0, got %s" % freq)
    if isinstance(phase, bool) or not isinstance(phase, (int, float)):
        raise TypeError(nmu.type_error_str(phase, "phase", "float"))
    return freq, float(phase)


def _validate_f0_f1(f0: float, f1: float) -> tuple[float, float]:
    if isinstance(f0, bool) or not isinstance(f0, (int, float)):
        raise TypeError(nmu.type_error_str(f0, "f0", "float"))
    f0 = float(f0)
    if f0 <= 0:
        raise ValueError("f0 must be > 0, got %s" % f0)
    if isinstance(f1, bool) or not isinstance(f1, (int, float)):
        raise TypeError(nmu.type_error_str(f1, "f1", "float"))
    f1 = float(f1)
    if f1 <= 0:
        raise ValueError("f1 must be > 0, got %s" % f1)
    return f0, f1


# =========================================================================
# Registry and factory
# =========================================================================

_PULSE_FUNC_REGISTRY: dict[str, type[NMPulseFunc]] = {
    "square": NMPulseFuncSquare,
    "ramp+":  NMPulseFuncRamp,
    "ramp-":  NMPulseFuncRamp,
    "exp":    NMPulseFuncExp,
    "alpha":  NMPulseFuncAlpha,
    "sin":    NMPulseFuncSin,
    "sinzap": NMPulseFuncSinZap,
    "user":   NMPulseFuncUser,
}


def _pulse_func_from_dict(d: dict | str) -> NMPulseFunc:
    """Create an :class:`NMPulseFunc` from a dict or bare shape-name string.

    Args:
        d: Dict with at least a ``"pulse"`` key, or a bare shape-name string.
            Additional keys are the shape-specific parameters for that
            subclass (e.g. ``"tau"`` for ``"exp"``, ``"freq"`` for
            ``"sin"``).  Unknown keys raise :exc:`KeyError`.

    Returns:
        An :class:`NMPulseFunc` subclass instance.

    Raises:
        TypeError: If *d* is not a dict or string.
        KeyError: If ``"pulse"`` key is missing, a required key is absent
            (e.g. ``"data"`` for ``"user"``), or an unknown key is present.
        ValueError: If the shape name is not in :data:`_VALID_SHAPES`, or a
            parameter value is invalid.
    """
    if isinstance(d, str):
        d = {"pulse": d}
    if not isinstance(d, dict):
        raise TypeError(nmu.type_error_str(d, "pulse func", "dictionary or string"))
    if "pulse" not in d:
        raise KeyError("missing key 'pulse'")
    name = d["pulse"]
    if not isinstance(name, str):
        raise TypeError(nmu.type_error_str(name, "pulse", "string"))
    name = name.lower()
    if name not in _PULSE_FUNC_REGISTRY:
        raise ValueError("unknown pulse shape %r; valid shapes: %s"
                         % (name, sorted(_VALID_SHAPES)))
    cls = _PULSE_FUNC_REGISTRY[name]

    # Extract valid keys for this subclass; reject unknown keys
    kwargs: dict = {}
    for key, val in d.items():
        k = key.lower()
        if k == "pulse":
            continue
        if k in cls._VALID_KEYS:
            kwargs[k] = val
        else:
            if cls._VALID_KEYS:
                raise KeyError(
                    "unknown key %r for pulse %r (valid: %s)"
                    % (key, name, sorted(cls._VALID_KEYS))
                )
            else:
                raise KeyError(
                    "pulse %r takes no extra parameters, got %r" % (name, key)
                )

    return cls(name=name, **kwargs)
