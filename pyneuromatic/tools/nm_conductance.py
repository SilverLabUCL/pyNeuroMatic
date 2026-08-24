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

    def voltage_factor(self, V: float) -> float:
        """Voltage-dependent scaling factor in [0, 1] (1.0 = no block).

        Override in subclasses with voltage-dependent gating (e.g. NMDA Mg²⁺
        block).  NMModelIAF.simulate() applies this to scale g_ext currents:
        I = g(t) × voltage_factor(V) × (V − e_rev).
        """
        return 1.0

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


class NMConductanceGABA(NMConductance):
    """GABAergic (Cl⁻) synaptic conductance — reversal potential register only.

    ``g_density=0.0``: the static (ohmic) contribution is zero.  The actual
    time-varying conductance ``g(t)`` is supplied externally as a pre-computed
    array (nS) via ``NMModel.simulate(g_ext=...)``.  This object exists
    solely to register ``e_rev`` in the conductance container so that
    ``simulate()`` can look up the reversal potential by name.

    Default ``e_rev = −70.0 mV`` (Cl⁻ reversal, GABAₐ).
    """

    def __init__(self, e_rev: float = -70.0, g_density: float = 0.0) -> None:
        super().__init__("gaba", g_density=0.0, e_rev=e_rev)


class NMConductanceAMPA(NMConductance):
    """AMPA-receptor synaptic conductance — reversal potential register only.

    Same design as :class:`NMConductanceGABA` (``g_density=0.0``; ``g(t)``
    supplied externally via ``NMModel.simulate(g_ext=...)``).

    Default ``e_rev = 0.0 mV`` (cation reversal, AMPA).
    """

    def __init__(self, e_rev: float = 0.0, g_density: float = 0.0) -> None:
        super().__init__("ampa", g_density=0.0, e_rev=e_rev)


class NMConductanceNMDA(NMConductance):
    """NMDA-receptor synaptic conductance with voltage-dependent Mg²⁺ block.

    ``g_density=0.0``: the static (ohmic) contribution is zero.  ``g(t)`` is
    supplied externally via ``NMModel.simulate(g_ext=...)``.

    :meth:`voltage_factor` returns the Mg²⁺ block factor B(V) for the
    selected model (``mg_block`` property).  NMModel scales each g_ext
    step as ``I = g(t) × B(V) × (V − e_rev)``.

    Block models:

    ``"boltzmann"`` (default)
        ``1 / (1 + exp(−(V − v_half) / v_slope))``
        Defaults: ``v_half=−12.8 mV``, ``v_slope=22.4 mV`` (Rothman 2009).
    ``"jahr_stevens_1990"``
        ``1 / (1 + mg_conc × exp(−0.062 V) / 3.57)``
        Default: ``mg_conc=1.0 mM``.
    ``"gc_schwartz_2012"``
        Multi-exponential fit (fixed constants from Igor NeuroMatic source).

    Default ``e_rev = 0.0 mV``.
    """

    _VALID_BLOCK_MODELS = frozenset({
        "boltzmann", "jahr_stevens_1990", "gc_schwartz_2012"
    })

    def __init__(
        self,
        e_rev: float = 0.0,
        g_density: float = 0.0,
        mg_block: str = "boltzmann",
        v_half: float = -12.8,
        v_slope: float = 22.4,
        mg_conc: float = 1.0,
    ) -> None:
        super().__init__("nmda", g_density=0.0, e_rev=e_rev)
        self.mg_block = mg_block
        self._v_half = float(v_half)
        self.v_slope = v_slope
        self.mg_conc = mg_conc

    @property
    def mg_block(self) -> str:
        """Mg²⁺ block model name."""
        return self._mg_block

    @mg_block.setter
    def mg_block(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("mg_block must be a string")
        if value not in self._VALID_BLOCK_MODELS:
            raise ValueError(
                "mg_block %r is not valid; choose one of %s"
                % (value, sorted(self._VALID_BLOCK_MODELS))
            )
        self._mg_block = value

    @property
    def v_half(self) -> float:
        """Boltzmann half-activation voltage (mV)."""
        return self._v_half

    @v_half.setter
    def v_half(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("v_half must be a float")
        self._v_half = float(value)

    @property
    def v_slope(self) -> float:
        """Boltzmann slope factor (mV); must be non-zero."""
        return self._v_slope

    @v_slope.setter
    def v_slope(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("v_slope must be a float")
        if value == 0:
            raise ValueError("v_slope must be non-zero")
        self._v_slope = float(value)

    @property
    def mg_conc(self) -> float:
        """Extracellular Mg²⁺ concentration (mM) for Jahr–Stevens model."""
        return self._mg_conc

    @mg_conc.setter
    def mg_conc(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("mg_conc must be a float")
        if value <= 0:
            raise ValueError("mg_conc must be > 0, got %g" % value)
        self._mg_conc = float(value)

    def voltage_factor(self, V: float) -> float:
        """Mg²⁺ block factor B(V) in [0, 1] for the selected block model."""
        if self._mg_block == "boltzmann":
            return 1.0 / (1.0 + math.exp(-(V - self._v_half) / self._v_slope))
        elif self._mg_block == "jahr_stevens_1990":
            return 1.0 / (1.0 + self._mg_conc * math.exp(-0.062 * V) / 3.57)
        else:  # gc_schwartz_2012
            e1 = math.exp((V - (-119.51)) / 38.427) + math.exp(-(V - (-45.895)) / 28.357)
            e2 = e1 + math.exp(-(V - 84.784) / 38.427)
            return e1 / e2

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "mg_block": self._mg_block,
            "v_half": self._v_half,
            "v_slope": self._v_slope,
            "mg_conc": self._mg_conc,
        })
        return d

    def __repr__(self) -> str:
        return "%s(e_rev=%g, mg_block=%r)" % (
            self.__class__.__name__, self._e_rev, self._mg_block
        )


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
        super().__init__("na_hh", g_density, e_rev)

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
        return {"conductance": "na_hh", "g_density": self._g_density, "e_rev": self._e_rev}

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
        super().__init__("k_hh", g_density, e_rev)

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
        return {"conductance": "k_hh", "g_density": self._g_density, "e_rev": self._e_rev}

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
# Rothman & Manis (2003) VCN conductances
# ──────────────────────────────────────────────────────────────────────────────

class NMConductanceVCNNa(NMConductance):
    """Rothman & Manis (2003) VCN fast sodium channel (m³h gating).

    Kinetics from Rothman & Manis (2003), J Neurophysiol 89: 3097–3113.
    Uses steady-state / time-constant (inf/tau) form:
        dm/dt = (m∞ − m) / τm;  dh/dt = (h∞ − h) / τh

    The ``dydt_scaled(q10)`` multiplier corresponds to ``TC = tau_q10^((T − 22)/10)``.

    Default: g_density = 0.75 nS/µm²  (= 1000 nS / 1333 µm²),  e_rev = 55 mV
    """

    def __init__(self, g_density: float = 0.75, e_rev: float = 55.0) -> None:
        super().__init__("na_vcn", g_density, e_rev)

    def n_states(self) -> int:
        return 2

    def gate_names(self) -> list[str]:
        return ["Na_m", "Na_h"]

    def state_init(self, V: float) -> list[float]:
        return [self._m_inf(V), self._h_inf(V)]

    def current(self, V: float, states: list[float]) -> float:
        m, h = states
        return self._g_density * (m ** 3) * h * (V - self._e_rev)

    def dydt(self, V: float, states: list[float]) -> list[float]:
        m, h = states
        return [
            (self._m_inf(V) - m) / self._tau_m(V),
            (self._h_inf(V) - h) / self._tau_h(V),
        ]

    def to_dict(self) -> dict:
        return {"conductance": "na_vcn", "g_density": self._g_density, "e_rev": self._e_rev}

    @staticmethod
    def _m_inf(V: float) -> float:
        return 1.0 / (1.0 + math.exp(-(V + 38.0) / 7.0))

    @staticmethod
    def _tau_m(V: float) -> float:
        v60 = V + 60.0
        return 0.04 + 10.0 / (5.0 * math.exp(v60 / 18.0) + 36.0 * math.exp(-v60 / 25.0))

    @staticmethod
    def _h_inf(V: float) -> float:
        return 1.0 / (1.0 + math.exp((V + 65.0) / 6.0))

    @staticmethod
    def _tau_h(V: float) -> float:
        v60 = V + 60.0
        return 0.6 + 100.0 / (7.0 * math.exp(v60 / 11.0) + 10.0 * math.exp(-v60 / 25.0))


class NMConductanceVCNKHT(NMConductance):
    """Rothman & Manis (2003) VCN high-threshold K channel (blended n²/p gating).

    Current formula:
        I = g × (0.85 × n² + 0.15 × p) × (V − E_K)

    where n and p are independent gating variables (K_n and K_p in the paper).

    Default: g_density = 0.1125 nS/µm²  (= 150 nS / 1333 µm²),  e_rev = −70 mV
    """

    _GHT_FRACTION: float = 0.85

    def __init__(self, g_density: float = 0.1125, e_rev: float = -70.0) -> None:
        super().__init__("kht_vcn", g_density, e_rev)

    def n_states(self) -> int:
        return 2

    def gate_names(self) -> list[str]:
        return ["K_n", "K_p"]

    def state_init(self, V: float) -> list[float]:
        return [self._n_inf(V), self._p_inf(V)]

    def current(self, V: float, states: list[float]) -> float:
        n, p = states
        f = self._GHT_FRACTION
        return self._g_density * (f * n ** 2 + (1.0 - f) * p) * (V - self._e_rev)

    def dydt(self, V: float, states: list[float]) -> list[float]:
        n, p = states
        return [
            (self._n_inf(V) - n) / self._tau_n(V),
            (self._p_inf(V) - p) / self._tau_p(V),
        ]

    def to_dict(self) -> dict:
        return {"conductance": "kht_vcn", "g_density": self._g_density, "e_rev": self._e_rev}

    @staticmethod
    def _n_inf(V: float) -> float:
        return (1.0 + math.exp(-(V + 15.0) / 5.0)) ** -0.5

    @staticmethod
    def _tau_n(V: float) -> float:
        v60 = V + 60.0
        return 0.7 + 100.0 / (11.0 * math.exp(v60 / 24.0) + 21.0 * math.exp(-v60 / 23.0))

    @staticmethod
    def _p_inf(V: float) -> float:
        return 1.0 / (1.0 + math.exp(-(V + 23.0) / 6.0))

    @staticmethod
    def _tau_p(V: float) -> float:
        v60 = V + 60.0
        return 5.0 + 100.0 / (4.0 * math.exp(v60 / 32.0) + 5.0 * math.exp(-v60 / 22.0))


class NMConductanceVCNKLT(NMConductance):
    """Rothman & Manis (2003) VCN low-threshold K channel — D-type (w⁴z gating).

    Current formula:
        I = g × w⁴ × z × (V − E_K)

    The z gate has a voltage-independent offset (z_fraction = 0.5):
        z∞ = 0.5 + 0.5 / (1 + exp((V + 71) / 10))

    Default: g_density = 0.0 nS/µm² (absent in I-c cell type),  e_rev = −70 mV
    """

    _Z_FRACTION: float = 0.5

    def __init__(self, g_density: float = 0.0, e_rev: float = -70.0) -> None:
        super().__init__("klt_vcn", g_density, e_rev)

    def n_states(self) -> int:
        return 2

    def gate_names(self) -> list[str]:
        return ["KD_w", "KD_z"]

    def state_init(self, V: float) -> list[float]:
        return [self._w_inf(V), self._z_inf(V)]

    def current(self, V: float, states: list[float]) -> float:
        w, z = states
        return self._g_density * (w ** 4) * z * (V - self._e_rev)

    def dydt(self, V: float, states: list[float]) -> list[float]:
        w, z = states
        return [
            (self._w_inf(V) - w) / self._tau_w(V),
            (self._z_inf(V) - z) / self._tau_z(V),
        ]

    def to_dict(self) -> dict:
        return {"conductance": "klt_vcn", "g_density": self._g_density, "e_rev": self._e_rev}

    @staticmethod
    def _w_inf(V: float) -> float:
        return (1.0 + math.exp(-(V + 48.0) / 6.0)) ** -0.25

    @staticmethod
    def _tau_w(V: float) -> float:
        v60 = V + 60.0
        return 1.5 + 100.0 / (6.0 * math.exp(v60 / 6.0) + 16.0 * math.exp(-v60 / 45.0))

    @classmethod
    def _z_inf(cls, V: float) -> float:
        zf = cls._Z_FRACTION
        return zf + (1.0 - zf) / (1.0 + math.exp((V + 71.0) / 10.0))

    @staticmethod
    def _tau_z(V: float) -> float:
        v60 = V + 60.0
        return 50.0 + 1000.0 / (math.exp(v60 / 20.0) + math.exp(-v60 / 8.0))


class NMConductanceVCNKA(NMConductance):
    """Rothman & Manis (2003) VCN transient A-type K channel (a⁴bc gating).

    Current formula:
        I = g × a⁴ × b × c × (V − E_KA)

    Gates b and c share the same steady state but have different time constants.

    Default: g_density = 0.0 nS/µm² (absent in I-c cell type),  e_rev = −70 mV
    """

    def __init__(self, g_density: float = 0.0, e_rev: float = -70.0) -> None:
        super().__init__("ka_vcn", g_density, e_rev)

    def n_states(self) -> int:
        return 3

    def gate_names(self) -> list[str]:
        return ["KA_a", "KA_b", "KA_c"]

    def state_init(self, V: float) -> list[float]:
        return [self._a_inf(V), self._b_inf(V), self._b_inf(V)]  # c shares b's steady state

    def current(self, V: float, states: list[float]) -> float:
        a, b, c = states
        return self._g_density * (a ** 4) * b * c * (V - self._e_rev)

    def dydt(self, V: float, states: list[float]) -> list[float]:
        a, b, c = states
        b_inf = self._b_inf(V)
        return [
            (self._a_inf(V) - a) / self._tau_a(V),
            (b_inf - b) / self._tau_b(V),
            (b_inf - c) / self._tau_c(V),
        ]

    def to_dict(self) -> dict:
        return {"conductance": "ka_vcn", "g_density": self._g_density, "e_rev": self._e_rev}

    @staticmethod
    def _a_inf(V: float) -> float:
        return (1.0 + math.exp(-(V + 31.0) / 6.0)) ** -0.25

    @staticmethod
    def _tau_a(V: float) -> float:
        v60 = V + 60.0
        return 0.1 + 100.0 / (7.0 * math.exp(v60 / 14.0) + 29.0 * math.exp(-v60 / 24.0))

    @staticmethod
    def _b_inf(V: float) -> float:
        return (1.0 + math.exp((V + 66.0) / 7.0)) ** -0.5

    @staticmethod
    def _tau_b(V: float) -> float:
        v60 = V + 60.0
        return 1.0 + 1000.0 / (14.0 * math.exp(v60 / 27.0) + 29.0 * math.exp(-v60 / 24.0))

    @staticmethod
    def _tau_c(V: float) -> float:
        return 10.0 + 90.0 / (1.0 + math.exp(-(V + 66.0) / 17.0))


class NMConductanceVCNH(NMConductance):
    """Rothman & Manis (2003) VCN hyperpolarisation-activated channel (r gating).

    Current formula:
        I = g × r × (V − E_H)

    Default: g_density = 3.75e-4 nS/µm²  (= 0.5 nS / 1333 µm²),  e_rev = −43 mV
    """

    def __init__(self, g_density: float = 3.75e-4, e_rev: float = -43.0) -> None:
        super().__init__("h_vcn", g_density, e_rev)

    def n_states(self) -> int:
        return 1

    def gate_names(self) -> list[str]:
        return ["H_r"]

    def state_init(self, V: float) -> list[float]:
        return [self._r_inf(V)]

    def current(self, V: float, states: list[float]) -> float:
        (r,) = states
        return self._g_density * r * (V - self._e_rev)

    def dydt(self, V: float, states: list[float]) -> list[float]:
        (r,) = states
        return [(self._r_inf(V) - r) / self._tau_r(V)]

    def to_dict(self) -> dict:
        return {"conductance": "h_vcn", "g_density": self._g_density, "e_rev": self._e_rev}

    @staticmethod
    def _r_inf(V: float) -> float:
        return 1.0 / (1.0 + math.exp((V + 76.0) / 7.0))

    @staticmethod
    def _tau_r(V: float) -> float:
        v60 = V + 60.0
        return 25.0 + 100000.0 / (237.0 * math.exp(v60 / 12.0) + 17.0 * math.exp(-v60 / 14.0))


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
    "leak":   NMConductanceLeak,
    "na_hh":   NMConductanceHHNa,
    "k_hh":    NMConductanceHHK,
    "na_vcn":  NMConductanceVCNNa,
    "kht_vcn":   NMConductanceVCNKHT,
    "klt_vcn": NMConductanceVCNKLT,
    "ka_vcn":  NMConductanceVCNKA,
    "h_vcn":   NMConductanceVCNH,
    "gaba":   NMConductanceGABA,
    "ampa":   NMConductanceAMPA,
    "nmda":   NMConductanceNMDA,
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
    return cls(**d)
