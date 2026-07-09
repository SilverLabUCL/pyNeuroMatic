# -*- coding: utf-8 -*-
"""
NMModel: neural ODE model classes.

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

from pyneuromatic.tools.nm_conductance import (
    NMConductanceLeak,
    NMConductanceHHNa,
    NMConductanceHHK,
    NMConductanceGABA,
    NMConductanceAMPA,
    NMConductanceNMDA,
    NMConductanceContainer,
    _conductance_from_dict,
)


class NMModel:
    """Abstract base class for neural simulation models.

    Subclasses implement :meth:`simulate` for a specific model type
    (e.g. Hodgkin–Huxley, integrate-and-fire).

    Common parameters:
        v0:          Resting membrane potential (mV).  Default −65.0.
        temperature: Simulation temperature (°C).     Default 6.3.
    """

    def __init__(
        self,
        name: str = "model",
        config: dict | None = None,
        nm_path: str = "model",
    ) -> None:
        self._name = name
        self._nm_path = nm_path
        self._v0: float = -65.0
        self._temperature: float = 6.3   # HH reference temperature (Hodgkin & Huxley 1952)

    @property
    def name(self) -> str:
        return self._name

    @property
    def v0(self) -> float:
        """Resting membrane potential (mV)."""
        return self._v0

    @v0.setter
    def v0(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("v0 must be a float")
        self._v0 = float(value)

    @property
    def temperature(self) -> float:
        """Simulation temperature (°C)."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("temperature must be a float")
        self._temperature = float(value)

    def simulate(
        self,
        n_points: int,
        xstart: float,
        xdelta: float,
        i_ext: np.ndarray,
        g_ext: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run a simulation and return state variable trajectories.

        Args:
            n_points: Number of time points.
            xstart:   Start time (ms).
            xdelta:   Time step (ms).
            i_ext:    External current waveform (pA), length ``n_points``.
            g_ext:    Optional synaptic conductance waveforms (nS), keyed by
                      conductance name (e.g. ``{"GABA": g_arr, "AMPA": g_arr}``).
                      Each array must have length ``n_points``.  The named
                      conductances must be registered in the model's conductance
                      container so that ``e_rev`` can be looked up.

        Returns:
            Dictionary mapping variable names to 1-D arrays of length
            ``n_points``.  Always contains ``"V"`` (mV).
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict, nm_path: str = "model") -> "NMModel":
        return _model_from_dict(d, nm_path=nm_path)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NMModel):
            return NotImplemented
        return self.to_dict() == other.to_dict()


# ──────────────────────────────────────────────────────────────────────────────
# Hodgkin–Huxley model
# ──────────────────────────────────────────────────────────────────────────────

class NMModelHH(NMModel):
    """Hodgkin–Huxley point-neuron model.

    Integrates the HH ODE system using
    :func:`scipy.integrate.solve_ivp` (RK45 by default).

    The neuron is modelled as a sphere.  Conductance densities are specified
    per unit area (nS/µm²); currents are scaled by the surface area before
    entering the ODE.

    Parameters:
        v0:          Resting potential (mV).              Default −65.0.
        temperature: Simulation temperature (°C).          Default 6.3.
        cm_density:  Specific membrane capacitance
                     (pF/µm²; 0.01 pF/µm² = 1 µF/cm²).   Default 0.01.
        diameter:    Cell diameter (µm).                   Default √(1200/π) ≈ 19.544
                     (gives SA = 1200 µm², Cm = 12 pF).
        tau_q10:     Q10 factor for HH rate constants.
                     Applied as ``tau_q10^((T − 6.3) / 10)``. Default 3.0.

    The default conductance set matches Hodgkin & Huxley (1952):
        Leak  g=0.003 nS/µm²  E=−54.4 mV
        Na    g=1.2   nS/µm²  E= 50.0 mV
        K     g=0.36  nS/µm²  E=−77.0 mV

    Units summary:
        time:        ms
        voltage:     mV
        current:     pA
        capacitance: pF  → dV/dt in mV/ms  (pA/pF = mV/ms)
    """

    _T_REF: float = 6.3  # °C, HH reference temperature
    # SA = π × d² = 1200 µm²  →  Cm = 0.01 × 1200 = 12 pF (round values)
    _DEFAULT_DIAMETER: float = math.sqrt(1200.0 / math.pi)

    def __init__(
        self,
        name: str = "hh",
        config: dict | None = None,
        nm_path: str = "model",
    ) -> None:
        super().__init__(name=name, nm_path=nm_path)
        self._cm_density: float = 0.01                    # pF/µm²
        self._diameter: float = self._DEFAULT_DIAMETER    # µm
        self._tau_q10: float = 3.0

        self._conductances = NMConductanceContainer(
            nm_path=nm_path + ".conductances"
        )
        self._conductances.add("Leak", NMConductanceLeak())
        self._conductances.add("Na",   NMConductanceHHNa())
        self._conductances.add("K",    NMConductanceHHK())

        if config is not None:
            self._config_set(config, quiet=True)

    # ------------------------------------------------------------------
    # Properties

    @property
    def cm_density(self) -> float:
        """Specific membrane capacitance (pF/µm²)."""
        return self._cm_density

    @cm_density.setter
    def cm_density(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("cm_density must be a float")
        if value <= 0:
            raise ValueError("cm_density must be > 0, got %g" % value)
        self._cm_density = float(value)

    @property
    def diameter(self) -> float:
        """Cell diameter (µm)."""
        return self._diameter

    @diameter.setter
    def diameter(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("diameter must be a float")
        if value <= 0:
            raise ValueError("diameter must be > 0, got %g" % value)
        self._diameter = float(value)

    @property
    def tau_q10(self) -> float:
        """Q10 factor applied to HH rate constants."""
        return self._tau_q10

    @tau_q10.setter
    def tau_q10(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("tau_q10 must be a float")
        if value <= 0:
            raise ValueError("tau_q10 must be > 0, got %g" % value)
        self._tau_q10 = float(value)

    @property
    def conductances(self) -> NMConductanceContainer:
        """The conductance container (Leak, Na, K by default)."""
        return self._conductances

    # ------------------------------------------------------------------
    # Derived quantities

    def _surface_area(self) -> float:
        """Sphere surface area in µm²: π × diameter²."""
        return math.pi * self._diameter ** 2

    def _capacitance(self) -> float:
        """Total membrane capacitance in pF."""
        return self._cm_density * self._surface_area()

    def _q10_factor(self) -> float:
        """Temperature scaling factor for HH rate constants."""
        return self._tau_q10 ** ((self._temperature - self._T_REF) / 10.0)

    # ------------------------------------------------------------------
    # Simulation

    def simulate(
        self,
        n_points: int,
        xstart: float,
        xdelta: float,
        i_ext: np.ndarray,
        g_ext: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Integrate the HH ODE system.

        Args:
            n_points: Number of time samples.
            xstart:   First time point (ms).
            xdelta:   Time step (ms; also used as ``max_step`` for solve_ivp).
            i_ext:    External current (pA), 1-D array of length ``n_points``.

        Returns:
            Dict with keys ``"V"`` (mV) plus one key per gate variable
            (``"m"``, ``"h"``, ``"n"``), each a 1-D array of length
            ``n_points``.
        """
        from scipy.integrate import solve_ivp

        if n_points < 1:
            raise ValueError("n_points must be >= 1")
        if xdelta <= 0:
            raise ValueError("xdelta must be > 0")
        i_ext = np.asarray(i_ext, dtype=float)
        if i_ext.shape != (n_points,):
            raise ValueError("i_ext must have length n_points (%d)" % n_points)

        SA = self._surface_area()
        Cm = self._capacitance()
        q10 = self._q10_factor()

        t = np.linspace(xstart, xstart + (n_points - 1) * xdelta, n_points)

        # Build state-vector offset map
        # y[0] = V;  y[offset : offset+n_states] = gates for each conductance
        offsets: dict[str, slice] = {}
        offset = 1
        for cname, cond in self._conductances:
            ns = cond.n_states()
            offsets[cname] = slice(offset, offset + ns)
            offset += ns
        n_y = offset

        # Initial conditions: steady-state gates at v0
        y0 = np.zeros(n_y)
        y0[0] = self._v0
        for cname, cond in self._conductances:
            sl = offsets[cname]
            init = cond.state_init(self._v0)
            if init:
                y0[sl] = init

        def ode(t_val: float, y: np.ndarray) -> np.ndarray:
            V = y[0]
            dydt = np.zeros(n_y)
            i_ionic = 0.0
            for cname, cond in self._conductances:
                sl = offsets[cname]
                states = list(y[sl])
                i_ionic += cond.current(V, states) * SA
                scaled = cond.dydt_scaled(V, states, q10)
                if scaled:
                    dydt[sl] = scaled
            t_idx = int((t_val - xstart) / xdelta)
            t_idx = max(0, min(t_idx, n_points - 1))
            dydt[0] = (i_ext[t_idx] - i_ionic) / Cm
            return dydt

        sol = solve_ivp(
            fun=ode,
            t_span=(t[0], t[-1]),
            y0=y0,
            method="RK45",
            t_eval=t,
            max_step=xdelta,
            dense_output=False,
        )

        result: dict[str, np.ndarray] = {"V": sol.y[0]}
        for cname, cond in self._conductances:
            sl = offsets[cname]
            for gate_idx, gate_name in enumerate(cond.gate_names()):
                result[gate_name] = sol.y[sl.start + gate_idx]

        return result

    # ------------------------------------------------------------------
    # Config / serialisation

    _SCALAR_KEYS = frozenset({"v0", "temperature", "cm_density", "diameter", "tau_q10"})

    def _config_set(self, config: dict, quiet: bool = True) -> None:
        """Apply a configuration dictionary (from ``to_dict()``)."""
        for key, value in config.items():
            if key == "model":
                continue
            if key == "conductances":
                self._conductances = NMConductanceContainer.from_dict(
                    {"conductances": value},
                    nm_path=self._nm_path + ".conductances",
                )
            elif key in self._SCALAR_KEYS:
                setattr(self, key, value)
            else:
                raise KeyError("unknown NMModelHH config key %r" % key)

    def to_dict(self) -> dict:
        d: dict = {
            "model": "hh",
            "v0": self._v0,
            "temperature": self._temperature,
            "cm_density": self._cm_density,
            "diameter": self._diameter,
            "tau_q10": self._tau_q10,
        }
        d["conductances"] = self._conductances.to_dict()["conductances"]
        return d

    @classmethod
    def from_dict(cls, d: dict, nm_path: str = "model") -> "NMModelHH":
        obj = cls(name=d.get("model", "hh"), nm_path=nm_path)
        obj._config_set(d, quiet=True)
        return obj

    def __repr__(self) -> str:
        return (
            "NMModelHH(v0=%g, cm_density=%g, diameter=%g, "
            "temperature=%g, tau_q10=%g, n_conductances=%d)"
            % (
                self._v0,
                self._cm_density,
                self._diameter,
                self._temperature,
                self._tau_q10,
                len(self._conductances),
            )
        )


# ──────────────────────────────────────────────────────────────────────────────
# Integrate-and-Fire model
# ──────────────────────────────────────────────────────────────────────────────

class NMModelIAF(NMModel):
    """Integrate-and-fire point-neuron model.

    When the membrane potential reaches ``ap_threshold``, a spike is inserted
    (the crossing point is set to ``ap_peak``) and the neuron is held at
    ``ap_reset`` for ``ap_refrac`` ms before integration resumes.

    The leak conductance lives in a :class:`NMConductanceContainer` so that
    synaptic conductances (GABA, AMPA, NMDA) can be added later without
    changing the core integration loop.

    Parameters:
        v0:           Resting / initial potential (mV).    Default −80.0.
        temperature:  Simulation temperature (°C).          Default 37.0.
                      Not currently used (no Q10-dependent kinetics).
        cm_density:   Specific membrane capacitance
                      (pF/µm²; 0.01 pF/µm² = 1 µF/cm²).   Default 0.01.
        diameter:     Cell diameter (µm).                   Default 10.0.
        ap_threshold: Spike threshold (mV).                 Default −40.0.
        ap_peak:      Spike peak inserted at threshold (mV).Default  32.0.
        ap_reset:     Post-spike reset potential (mV).       Default −61.0.
        ap_refrac:    Absolute refractory period (ms; > 0). Default 2.0.
        method:       Integration method: ``"exact"`` (default) or
                      ``"euler"``.

    Integration methods:

    ``"exact"`` — closed-form update between spikes:

        V(t+dt) = V_ss + (V(t) − V_ss) × exp(−dt / τ_m)

    where V_ss = (I_ext + Σ gᵢ·SAᵢ·Eᵢ) / G_total and τ_m = Cm / G_total.
    ``exp(−dt/τ_m)`` is precomputed once per simulation.  Correct at any
    step size; allows larger ``xdelta`` for faster runs at equal accuracy.
    Assumes all conductances are ohmic (no time-varying gating) — use
    ``"euler"`` once synaptic conductances with kinetics are added.

    ``"euler"`` — forward Euler: V(t+dt) = V(t) + dV/dt · dt.  Simple but
    accumulates O(dt) error per step (ISI error ≈ 0.1 ms at dt = 0.025 ms).

    Default conductance matches the Igor NeuroMatic IAF reference cell:
        Leak  g = 0.9 nS total / SA ≈ 0.002865 nS/µm²,  E = −80 mV
        (0.9 nS absorbs the tonic-GABA contribution; split out when GABA is added.)

    Units summary:
        time:        ms
        voltage:     mV
        current:     pA
        capacitance: pF  → dV/dt in mV/ms  (pA/pF = mV/ms)
    """

    _DEFAULT_DIAMETER:       float = 10.0
    # 0.9 nS total / (π × 10²) µm²
    _DEFAULT_LEAK_G_DENSITY: float = 0.9 / (math.pi * 10.0 ** 2)
    _DEFAULT_LEAK_E_REV:     float = -80.0
    _VALID_METHODS = frozenset({"euler", "exact"})

    def __init__(
        self,
        name: str = "iaf",
        config: dict | None = None,
        nm_path: str = "model",
    ) -> None:
        super().__init__(name=name, nm_path=nm_path)
        self._v0:           float = -80.0
        self._temperature:  float = 37.0
        self._cm_density:   float = 0.01
        self._diameter:     float = self._DEFAULT_DIAMETER
        self._ap_threshold: float = -40.0
        self._ap_peak:      float =  32.0
        self._ap_reset:     float = -61.0
        self._ap_refrac:    float =   2.0
        self._method:       str   = "exact"

        self._conductances = NMConductanceContainer(
            nm_path=nm_path + ".conductances"
        )
        self._conductances.add(
            "Leak",
            NMConductanceLeak(
                g_density=self._DEFAULT_LEAK_G_DENSITY,
                e_rev=self._DEFAULT_LEAK_E_REV,
            ),
        )

        if config is not None:
            self._config_set(config, quiet=True)

    # ------------------------------------------------------------------
    # Properties

    @property
    def cm_density(self) -> float:
        """Specific membrane capacitance (pF/µm²)."""
        return self._cm_density

    @cm_density.setter
    def cm_density(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("cm_density must be a float")
        if value <= 0:
            raise ValueError("cm_density must be > 0, got %g" % value)
        self._cm_density = float(value)

    @property
    def diameter(self) -> float:
        """Cell diameter (µm)."""
        return self._diameter

    @diameter.setter
    def diameter(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("diameter must be a float")
        if value <= 0:
            raise ValueError("diameter must be > 0, got %g" % value)
        self._diameter = float(value)

    @property
    def ap_threshold(self) -> float:
        """Spike threshold (mV)."""
        return self._ap_threshold

    @ap_threshold.setter
    def ap_threshold(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("ap_threshold must be a float")
        self._ap_threshold = float(value)

    @property
    def ap_peak(self) -> float:
        """Spike peak potential inserted at threshold crossing (mV)."""
        return self._ap_peak

    @ap_peak.setter
    def ap_peak(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("ap_peak must be a float")
        self._ap_peak = float(value)

    @property
    def ap_reset(self) -> float:
        """Post-spike reset potential (mV)."""
        return self._ap_reset

    @ap_reset.setter
    def ap_reset(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("ap_reset must be a float")
        self._ap_reset = float(value)

    @property
    def ap_refrac(self) -> float:
        """Absolute refractory period (ms)."""
        return self._ap_refrac

    @ap_refrac.setter
    def ap_refrac(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("ap_refrac must be a float")
        if value <= 0:
            raise ValueError("ap_refrac must be > 0, got %g" % value)
        self._ap_refrac = float(value)

    @property
    def method(self) -> str:
        """Integration method: ``"exact"`` (default) or ``"euler"``."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("method must be a string")
        if value not in self._VALID_METHODS:
            raise ValueError(
                "method must be one of %s, got %r"
                % (sorted(self._VALID_METHODS), value)
            )
        self._method = value

    @property
    def conductances(self) -> NMConductanceContainer:
        """The conductance container (Leak by default)."""
        return self._conductances

    # ------------------------------------------------------------------
    # Derived quantities

    def _surface_area(self) -> float:
        """Sphere surface area in µm²: π × diameter²."""
        return math.pi * self._diameter ** 2

    def _capacitance(self) -> float:
        """Total membrane capacitance in pF."""
        return self._cm_density * self._surface_area()

    # ------------------------------------------------------------------
    # Simulation

    def simulate(
        self,
        n_points: int,
        xstart: float,
        xdelta: float,
        i_ext: np.ndarray,
        g_ext: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Integrate the IAF ODE with threshold detection.

        Args:
            n_points: Number of time samples.
            xstart:   First time point (ms).  Not used in integration but
                      stored for API consistency with NMModelHH.
            xdelta:   Time step (ms; also determines refractory step count).
            i_ext:    External current (pA), 1-D array of length ``n_points``.
            g_ext:    Optional synaptic conductance waveforms (nS), keyed by
                      the conductance name as registered in
                      :attr:`conductances` (e.g. ``{"GABA": g_arr}``).
                      Each array must have length ``n_points``.  The named
                      conductances must already be in the container so that
                      ``e_rev`` can be looked up.

        Returns:
            Dict with key ``"V"`` (mV), a 1-D array of length ``n_points``.
            Spike peaks are recorded at the threshold-crossing sample;
            refractory samples are held at ``ap_reset``.
        """
        if n_points < 1:
            raise ValueError("n_points must be >= 1")
        if xdelta <= 0:
            raise ValueError("xdelta must be > 0")
        i_ext = np.asarray(i_ext, dtype=float)
        if i_ext.shape != (n_points,):
            raise ValueError("i_ext must have length n_points (%d)" % n_points)
        if g_ext:
            g_ext = {name: np.asarray(arr, dtype=float) for name, arr in g_ext.items()}
            for name, g_arr in g_ext.items():
                if name not in self._conductances:
                    raise KeyError("g_ext key %r not found in conductances" % name)
                if g_arr.shape != (n_points,):
                    raise ValueError(
                        "g_ext[%r] must have length %d" % (name, n_points)
                    )

        SA = self._surface_area()
        Cm = self._capacitance()
        refrac_steps = max(1, round(self._ap_refrac / xdelta))

        # Synaptic e_rev lookup — avoid repeated dict access inside the loop
        syn_e_revs = (
            {name: self._conductances[name].e_rev for name in g_ext}
            if g_ext else {}
        )

        V = np.zeros(n_points)
        V[0] = self._v0
        refrac_remaining = 0

        if self._method == "exact":
            # Static (ohmic) conductances precomputed once.
            # g_density=0 for GABA/AMPA, so they contribute nothing here.
            G_static      = sum(cond.g_density * SA for _, cond in self._conductances)
            sum_gE_static = sum(
                cond.g_density * SA * cond.e_rev for _, cond in self._conductances
            )

            for i in range(n_points - 1):
                if refrac_remaining > 0:
                    V[i + 1] = self._ap_reset
                    refrac_remaining -= 1
                    continue
                v = V[i]
                # Per-step: fold in time-varying synaptic conductances
                G_total = G_static
                sum_gE  = sum_gE_static
                if g_ext:
                    for name, g_arr in g_ext.items():
                        g_i   = g_arr[i] * self._conductances[name].voltage_factor(v)
                        G_total += g_i
                        sum_gE  += g_i * syn_e_revs[name]
                tau_m      = Cm / G_total
                exp_factor = math.exp(-xdelta / tau_m)
                V_ss       = (i_ext[i] + sum_gE) / G_total
                v_next     = V_ss + (v - V_ss) * exp_factor
                if v_next >= self._ap_threshold:
                    V[i] = self._ap_peak
                    V[i + 1] = self._ap_reset
                    refrac_remaining = refrac_steps - 1
                else:
                    V[i + 1] = v_next
        else:
            for i in range(n_points - 1):
                if refrac_remaining > 0:
                    V[i + 1] = self._ap_reset
                    refrac_remaining -= 1
                    continue
                v = V[i]
                i_ionic = sum(
                    cond.current(v, []) * SA for _, cond in self._conductances
                )
                if g_ext:
                    for name, g_arr in g_ext.items():
                        i_ionic += (
                            g_arr[i]
                            * self._conductances[name].voltage_factor(v)
                            * (v - syn_e_revs[name])
                        )
                v_next = v + (i_ext[i] - i_ionic) / Cm * xdelta
                if v_next >= self._ap_threshold:
                    V[i] = self._ap_peak
                    V[i + 1] = self._ap_reset
                    refrac_remaining = refrac_steps - 1
                else:
                    V[i + 1] = v_next

        return {"V": V}

    # ------------------------------------------------------------------
    # Config / serialisation

    _SCALAR_KEYS = frozenset({
        "v0", "temperature", "cm_density", "diameter",
        "ap_threshold", "ap_peak", "ap_reset", "ap_refrac",
    })

    def _config_set(self, config: dict, quiet: bool = True) -> None:
        for key, value in config.items():
            if key == "model":
                continue
            if key == "conductances":
                self._conductances = NMConductanceContainer.from_dict(
                    {"conductances": value},
                    nm_path=self._nm_path + ".conductances",
                )
            elif key == "method":
                self.method = value
            elif key in self._SCALAR_KEYS:
                setattr(self, key, value)
            else:
                raise KeyError("unknown NMModelIAF config key %r" % key)

    def to_dict(self) -> dict:
        d: dict = {
            "model":        "iaf",
            "v0":           self._v0,
            "temperature":  self._temperature,
            "cm_density":   self._cm_density,
            "diameter":     self._diameter,
            "ap_threshold": self._ap_threshold,
            "ap_peak":      self._ap_peak,
            "ap_reset":     self._ap_reset,
            "ap_refrac":    self._ap_refrac,
            "method":       self._method,
        }
        d["conductances"] = self._conductances.to_dict()["conductances"]
        return d

    @classmethod
    def from_dict(cls, d: dict, nm_path: str = "model") -> "NMModelIAF":
        obj = cls(name=d.get("model", "iaf"), nm_path=nm_path)
        obj._config_set(d, quiet=True)
        return obj

    def __repr__(self) -> str:
        return (
            "NMModelIAF(v0=%g, cm_density=%g, diameter=%g, "
            "ap_threshold=%g, ap_peak=%g, ap_reset=%g, ap_refrac=%g, method=%r)"
            % (
                self._v0,
                self._cm_density,
                self._diameter,
                self._ap_threshold,
                self._ap_peak,
                self._ap_reset,
                self._ap_refrac,
                self._method,
            )
        )


# ──────────────────────────────────────────────────────────────────────────────
# Registry and factory
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, type[NMModel]] = {
    "hh":  NMModelHH,
    "iaf": NMModelIAF,
}


def _model_from_dict(d: dict, nm_path: str = "model") -> NMModel:
    """Construct an :class:`NMModel` from a ``to_dict()`` dictionary."""
    mtype = d.get("model")
    if mtype not in _MODEL_REGISTRY:
        raise KeyError(
            "unknown model type %r; valid types: %s"
            % (mtype, sorted(_MODEL_REGISTRY))
        )
    return _MODEL_REGISTRY[mtype].from_dict(d, nm_path=nm_path)
