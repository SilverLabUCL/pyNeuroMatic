# -*- coding: utf-8 -*-
"""
NMConductance: ion channel conductance classes for neural ODE models.

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


class NMConductance:
    """Base class for ion channel conductance models.

    Subclasses implement specific channel kinetics (gating variables,
    alpha/beta rate functions). Each conductance is parameterised by its
    maximum conductance density (``g_density``, nS/µm²) and reversal
    potential (``e_rev``, mV).

    Units used throughout:
        - conductance density: nS/µm²
        - reversal potential:  mV
        - current density:     pA/µm²  (nS/µm² × mV = pA/µm²)
        - current:             pA      (multiply density by surface area µm²)
    """

    def __init__(self, name: str, g_density: float, e_rev: float) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if isinstance(g_density, bool) or not isinstance(g_density, (int, float)):
            raise TypeError("g_density must be a float")
        if g_density < 0:
            raise ValueError("g_density must be >= 0, got %g" % g_density)
        if isinstance(e_rev, bool) or not isinstance(e_rev, (int, float)):
            raise TypeError("e_rev must be a float")
        self._name = name
        self._g_density = float(g_density)
        self._e_rev = float(e_rev)

    @property
    def name(self) -> str:
        """Conductance type name (read-only)."""
        return self._name

    @property
    def g_density(self) -> float:
        """Maximum conductance density in nS/µm²."""
        return self._g_density

    @g_density.setter
    def g_density(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("g_density must be a float")
        if value < 0:
            raise ValueError("g_density must be >= 0, got %g" % value)
        self._g_density = float(value)

    @property
    def e_rev(self) -> float:
        """Reversal potential in mV."""
        return self._e_rev

    @e_rev.setter
    def e_rev(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("e_rev must be a float")
        self._e_rev = float(value)

    # ------------------------------------------------------------------
    # Channel interface — override in subclasses

    def n_states(self) -> int:
        """Number of gating variables (0 for ohmic leak)."""
        return 0

    def gate_names(self) -> list[str]:
        """Names of gating variables in order (e.g. ['m', 'h'] for Na)."""
        return []

    def state_init(self, V: float) -> list[float]:
        """Steady-state gating variable values at membrane voltage V (mV)."""
        return []

    def current(self, V: float, states: list[float]) -> float:
        """Conductance current density in pA/µm² at voltage V (mV).

        Multiply by surface area (µm²) to get current in pA.
        """
        return self._g_density * (V - self._e_rev)

    def dydt(self, V: float, states: list[float]) -> list[float]:
        """Gate variable derivatives at the HH reference temperature (6.3°C)."""
        return []

    def dydt_scaled(self, V: float, states: list[float], q10: float) -> list[float]:
        """Gate variable derivatives scaled by Q10 temperature factor."""
        return [r * q10 for r in self.dydt(V, states)]

    # ------------------------------------------------------------------
    # Serialisation

    def to_dict(self) -> dict:
        return {
            "conductance": self._name,
            "g_density": self._g_density,
            "e_rev": self._e_rev,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NMConductance":
        return _conductance_from_dict(d)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMConductance):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return "%s(g_density=%g, e_rev=%g)" % (
            self.__class__.__name__,
            self._g_density,
            self._e_rev,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Concrete conductances
# ──────────────────────────────────────────────────────────────────────────────

class NMConductanceLeak(NMConductance):
    """Ohmic leak conductance (no gating variables).

    I_L = g_density * (V − e_rev)

    Default values match Hodgkin & Huxley (1952):
        g_density = 0.003 nS/µm²  (= 0.3 mS/cm²)
        e_rev     = −54.387 mV  (= −65 + 10.613, the exact HH 1952 value)
    """

    def __init__(self, g_density: float = 0.003, e_rev: float = -54.387) -> None:
        super().__init__("leak", g_density, e_rev)


class NMConductanceHHNa(NMConductance):
    """Hodgkin–Huxley sodium channel (m³h gating).

    I_Na = g_density * m³ * h * (V − e_rev)

    Rate functions from Hodgkin & Huxley (1952), shifted to
    V_rest = −65 mV convention.

    Default values:
        g_density = 1.2  nS/µm²  (= 120 mS/cm²)
        e_rev     = 50.0 mV
    """

    def __init__(self, g_density: float = 1.2, e_rev: float = 50.0) -> None:
        super().__init__("hhna", g_density, e_rev)

    def n_states(self) -> int:
        return 2

    def gate_names(self) -> list[str]:
        return ["m", "h"]

    def state_init(self, V: float) -> list[float]:
        am, bm = self._alpha_m(V), self._beta_m(V)
        ah, bh = self._alpha_h(V), self._beta_h(V)
        m_inf = am / (am + bm)
        h_inf = ah / (ah + bh)
        return [m_inf, h_inf]

    def current(self, V: float, states: list[float]) -> float:
        m, h = states
        return self._g_density * (m ** 3) * h * (V - self._e_rev)

    def dydt(self, V: float, states: list[float]) -> list[float]:
        m, h = states
        dm = self._alpha_m(V) * (1.0 - m) - self._beta_m(V) * m
        dh = self._alpha_h(V) * (1.0 - h) - self._beta_h(V) * h
        return [dm, dh]

    def to_dict(self) -> dict:
        return {"conductance": "hhna", "g_density": self._g_density, "e_rev": self._e_rev}

    # ------------------------------------------------------------------
    # Rate functions (HH 1952, V_rest = −65 mV convention)

    @staticmethod
    def _alpha_m(V: float) -> float:
        dv = V + 40.0
        if abs(dv) < 1e-7:
            return 1.0
        return 0.1 * dv / (1.0 - math.exp(-dv / 10.0))

    @staticmethod
    def _beta_m(V: float) -> float:
        return 4.0 * math.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def _alpha_h(V: float) -> float:
        return 0.07 * math.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def _beta_h(V: float) -> float:
        return 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))


class NMConductanceHHK(NMConductance):
    """Hodgkin–Huxley delayed-rectifier potassium channel (n⁴ gating).

    I_K = g_density * n⁴ * (V − e_rev)

    Rate functions from Hodgkin & Huxley (1952), shifted to
    V_rest = −65 mV convention.

    Default values:
        g_density = 0.36 nS/µm²  (= 36 mS/cm²)
        e_rev     = −77.0 mV
    """

    def __init__(self, g_density: float = 0.36, e_rev: float = -77.0) -> None:
        super().__init__("hhk", g_density, e_rev)

    def n_states(self) -> int:
        return 1

    def gate_names(self) -> list[str]:
        return ["n"]

    def state_init(self, V: float) -> list[float]:
        an, bn = self._alpha_n(V), self._beta_n(V)
        n_inf = an / (an + bn)
        return [n_inf]

    def current(self, V: float, states: list[float]) -> float:
        (n,) = states
        return self._g_density * (n ** 4) * (V - self._e_rev)

    def dydt(self, V: float, states: list[float]) -> list[float]:
        (n,) = states
        dn = self._alpha_n(V) * (1.0 - n) - self._beta_n(V) * n
        return [dn]

    def to_dict(self) -> dict:
        return {"conductance": "hhk", "g_density": self._g_density, "e_rev": self._e_rev}

    # ------------------------------------------------------------------
    # Rate functions

    @staticmethod
    def _alpha_n(V: float) -> float:
        dv = V + 55.0
        if abs(dv) < 1e-7:
            return 0.1
        return 0.01 * dv / (1.0 - math.exp(-dv / 10.0))

    @staticmethod
    def _beta_n(V: float) -> float:
        return 0.125 * math.exp(-(V + 65.0) / 80.0)


# ──────────────────────────────────────────────────────────────────────────────
# Container
# ──────────────────────────────────────────────────────────────────────────────

class NMConductanceContainer:
    """Ordered collection of named :class:`NMConductance` objects.

    Analogous to :class:`~pyneuromatic.tools.nm_pulse.NMPulseContainer`.
    """

    def __init__(self, nm_path: str = "model.conductances") -> None:
        self._conductances: dict[str, NMConductance] = {}
        self._nm_path = nm_path

    def add(self, name: str, conductance: NMConductance) -> NMConductance:
        """Add a named conductance, replacing any existing entry with the same name."""
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(conductance, NMConductance):
            raise TypeError("conductance must be an NMConductance instance")
        self._conductances[name] = conductance
        return conductance

    def __getitem__(self, name: str) -> NMConductance:
        return self._conductances[name]

    def __iter__(self):
        return iter(self._conductances.items())

    def __len__(self) -> int:
        return len(self._conductances)

    def __contains__(self, name: str) -> bool:
        return name in self._conductances

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMConductanceContainer):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def to_dict(self) -> dict:
        return {
            "conductances": [
                {"name": name, **cond.to_dict()}
                for name, cond in self._conductances.items()
            ]
        }

    @classmethod
    def from_dict(cls, d: dict, nm_path: str = "model.conductances") -> "NMConductanceContainer":
        container = cls(nm_path=nm_path)
        for entry in d.get("conductances", []):
            entry = dict(entry)
            name = entry.pop("name")
            cond = _conductance_from_dict(entry)
            container.add(name, cond)
        return container


# ──────────────────────────────────────────────────────────────────────────────
# Registry and factory
# ──────────────────────────────────────────────────────────────────────────────

_CONDUCTANCE_REGISTRY: dict[str, type[NMConductance]] = {
    "leak": NMConductanceLeak,
    "hhna": NMConductanceHHNa,
    "hhk":  NMConductanceHHK,
}


def _conductance_from_dict(d: dict) -> NMConductance:
    """Construct an :class:`NMConductance` from a ``to_dict()`` dictionary."""
    d = dict(d)
    ctype = d.pop("conductance", None)
    if ctype not in _CONDUCTANCE_REGISTRY:
        raise KeyError(
            "unknown conductance type %r; valid types: %s"
            % (ctype, sorted(_CONDUCTANCE_REGISTRY))
        )
    cls = _CONDUCTANCE_REGISTRY[ctype]
    kwargs: dict = {}
    if "g_density" in d:
        kwargs["g_density"] = d["g_density"]
    if "e_rev" in d:
        kwargs["e_rev"] = d["e_rev"]
    return cls(**kwargs)
