# -*- coding: utf-8 -*-
"""Tests for NMModel and NMModelHH."""
import math
import numpy as np
import pytest

from pyneuromatic.tools.nm_model import NMModel, NMModelHH, _model_from_dict
from pyneuromatic.tools.nm_conductance import (
    NMConductanceLeak,
    NMConductanceHHNa,
    NMConductanceHHK,
)
from pyneuromatic.tools.nm_pulse import NMPulseContainer

# Simulation parameters used across many tests
N_POINTS = 4000   # 100 ms at 0.025 ms/step
XSTART = 0.0      # ms
XDELTA = 0.025    # ms
I_ONSET_IDX = 200  # sample index where step begins (= 5 ms)
I_AMP_SUPRA = 300.0   # pA — well above threshold for 20 µm HH sphere


def _make_i_ext(n_points, onset_idx, amp, duration_idx=800):
    """Build a square current step."""
    i_ext = np.zeros(n_points)
    end_idx = min(n_points, onset_idx + duration_idx)
    i_ext[onset_idx:end_idx] = amp
    return i_ext


def _sim_default(i_amp=I_AMP_SUPRA):
    """Run one default HH simulation with a suprathreshold step."""
    m = NMModelHH()
    i_ext = _make_i_ext(N_POINTS, I_ONSET_IDX, i_amp)
    return m.simulate(N_POINTS, XSTART, XDELTA, i_ext)


# ──────────────────────────────────────────────────────────────────────────────
# NMModelHH construction
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelHHConstruct:
    def test_default_params(self):
        m = NMModelHH()
        assert m.v0 == pytest.approx(-65.0)
        assert m.temperature == pytest.approx(6.3)   # HH reference temperature
        assert m.cm_density == pytest.approx(0.01)
        assert m.diameter == pytest.approx(math.sqrt(1200.0 / math.pi))
        assert m.tau_q10 == pytest.approx(3.0)

    def test_default_conductances(self):
        m = NMModelHH()
        assert "Leak" in m.conductances
        assert "Na"   in m.conductances
        assert "K"    in m.conductances
        assert len(m.conductances) == 3

    def test_conductance_types(self):
        m = NMModelHH()
        assert isinstance(m.conductances["Leak"], NMConductanceLeak)
        assert isinstance(m.conductances["Na"],   NMConductanceHHNa)
        assert isinstance(m.conductances["K"],    NMConductanceHHK)

    def test_name(self):
        m = NMModelHH(name="test_hh")
        assert m.name == "test_hh"

    def test_v0_setter(self):
        m = NMModelHH()
        m.v0 = -70.0
        assert m.v0 == pytest.approx(-70.0)

    def test_v0_setter_bool_raises(self):
        m = NMModelHH()
        with pytest.raises(TypeError):
            m.v0 = True

    def test_cm_density_setter_zero_raises(self):
        m = NMModelHH()
        with pytest.raises(ValueError):
            m.cm_density = 0.0

    def test_diameter_setter_zero_raises(self):
        m = NMModelHH()
        with pytest.raises(ValueError):
            m.diameter = 0.0

    def test_tau_q10_setter_zero_raises(self):
        m = NMModelHH()
        with pytest.raises(ValueError):
            m.tau_q10 = 0.0

    def test_config_kwarg(self):
        config = {"v0": -70.0, "diameter": 30.0}
        m = NMModelHH(config=config)
        assert m.v0 == pytest.approx(-70.0)
        assert m.diameter == pytest.approx(30.0)

    def test_config_unknown_key_raises(self):
        m = NMModelHH()
        with pytest.raises(KeyError):
            m._config_set({"nonexistent_key": 1.0})


# ──────────────────────────────────────────────────────────────────────────────
# simulate() return shape and keys
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelHHSimulateShape:
    def test_returns_dict(self):
        result = _sim_default()
        assert isinstance(result, dict)

    def test_has_V_key(self):
        result = _sim_default()
        assert "V" in result

    def test_has_gate_keys(self):
        result = _sim_default()
        assert "m" in result
        assert "h" in result
        assert "n" in result

    def test_array_lengths(self):
        result = _sim_default()
        for key, arr in result.items():
            assert len(arr) == N_POINTS, "key %r length mismatch" % key

    def test_invalid_n_points(self):
        m = NMModelHH()
        with pytest.raises(ValueError):
            m.simulate(0, 0.0, 0.025, np.zeros(0))

    def test_invalid_xdelta(self):
        m = NMModelHH()
        with pytest.raises(ValueError):
            m.simulate(100, 0.0, 0.0, np.zeros(100))

    def test_wrong_i_ext_length(self):
        m = NMModelHH()
        with pytest.raises(ValueError):
            m.simulate(100, 0.0, 0.025, np.zeros(50))


# ──────────────────────────────────────────────────────────────────────────────
# Resting state stability
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelHHRest:
    def test_equilibrium_potential(self):
        """True HH resting potential (zero-current root) should be ≈ −64.9964 mV."""
        from scipy.optimize import brentq
        m = NMModelHH()
        def total_i(V):
            return sum(
                cond.current(V, cond.state_init(V))
                for _, cond in m.conductances
            )
        V_rest = brentq(total_i, -80, -40, xtol=1e-12)
        assert V_rest == pytest.approx(-64.996376, abs=1e-4)

    def test_resting_potential_stable(self):
        """Zero input: Vm should stay near v0."""
        m = NMModelHH()
        i_ext = np.zeros(N_POINTS)
        result = m.simulate(N_POINTS, XSTART, XDELTA, i_ext)
        V = result["V"]
        assert np.all(np.abs(V - m.v0) < 2.0), (
            "Vm drifted > 2 mV from rest with no input"
        )

    def test_gates_near_steady_state_at_rest(self):
        """Gate variables should stay near their steady-state values."""
        m = NMModelHH()
        i_ext = np.zeros(N_POINTS)
        result = m.simulate(N_POINTS, XSTART, XDELTA, i_ext)
        m_inf, h_inf = m.conductances["Na"].state_init(m.v0)
        (n_inf,) = m.conductances["K"].state_init(m.v0)
        assert np.all(np.abs(result["m"] - m_inf) < 0.01)
        assert np.all(np.abs(result["h"] - h_inf) < 0.01)
        assert np.all(np.abs(result["n"] - n_inf) < 0.01)


# ──────────────────────────────────────────────────────────────────────────────
# Action potential generation
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelHHActionPotential:
    def test_ap_peak_above_zero(self):
        """Suprathreshold current step should produce AP with peak > 0 mV."""
        result = _sim_default()
        assert result["V"].max() > 0.0

    def test_ap_starts_near_rest(self):
        """Vm before current onset should be near v0."""
        m = NMModelHH()
        i_ext = _make_i_ext(N_POINTS, I_ONSET_IDX, I_AMP_SUPRA)
        result = m.simulate(N_POINTS, XSTART, XDELTA, i_ext)
        pre_onset_V = result["V"][:I_ONSET_IDX]
        assert np.all(np.abs(pre_onset_V - m.v0) < 2.0)

    def test_ap_peak_after_onset(self):
        """Peak should occur after the current step onset."""
        result = _sim_default()
        V = result["V"]
        peak_idx = int(np.argmax(V))
        assert peak_idx > I_ONSET_IDX

    def test_ap_peak_within_20ms_of_onset(self):
        """Peak should occur within 20 ms after step onset (AP is fast)."""
        result = _sim_default()
        V = result["V"]
        peak_idx = int(np.argmax(V))
        peak_time_after_onset = (peak_idx - I_ONSET_IDX) * XDELTA
        assert peak_time_after_onset < 20.0

    def test_ap_returns_to_rest(self):
        """After a short step, Vm should return within ±10 mV of v0."""
        m = NMModelHH()
        # Short 1 ms step, long simulation (200 ms)
        n_pts = 8000
        i_ext = _make_i_ext(n_pts, I_ONSET_IDX, I_AMP_SUPRA, duration_idx=40)
        result = m.simulate(n_pts, XSTART, XDELTA, i_ext)
        V = result["V"]
        # Check last 50 ms (last 2000 samples)
        v_end = V[-2000:]
        assert np.all(np.abs(v_end - m.v0) < 10.0), (
            "Vm did not return to rest after AP: max deviation = %g mV"
            % np.max(np.abs(v_end - m.v0))
        )

    def test_m_gate_activates_during_ap(self):
        """m gate should open significantly during the AP."""
        result = _sim_default()
        assert result["m"].max() > 0.5

    def test_h_gate_inactivates_during_ap(self):
        """h gate should inactivate (decrease) during the AP."""
        m = NMModelHH()
        i_ext = _make_i_ext(N_POINTS, I_ONSET_IDX, I_AMP_SUPRA)
        result = m.simulate(N_POINTS, XSTART, XDELTA, i_ext)
        h = result["h"]
        # h should be significantly lower at some point during the AP
        assert h.min() < 0.3

    def test_n_gate_activates_during_repolarisation(self):
        """n gate should open during the AP for K repolarisation."""
        result = _sim_default()
        assert result["n"].max() > 0.5

    def test_subthreshold_no_ap(self):
        """Subthreshold stimulus should not produce an AP."""
        m = NMModelHH()
        # Very small current
        i_ext = _make_i_ext(N_POINTS, I_ONSET_IDX, amp=1.0)
        result = m.simulate(N_POINTS, XSTART, XDELTA, i_ext)
        assert result["V"].max() < 0.0

    @pytest.mark.parametrize("i_amp,expected_aps", [
        (25,  0),
        (30,  1),
        (100, 7),
        (200, 9),
    ])
    def test_ap_count(self, i_amp, expected_aps):
        """AP count for 100 ms square step after 25 ms settling period."""
        from scipy.signal import find_peaks
        xdelta = 0.025
        onset_idx = round(25.0 / xdelta)
        end_idx = onset_idx + round(100.0 / xdelta)
        n_pts = round(150.0 / xdelta)
        i_ext = np.zeros(n_pts)
        i_ext[onset_idx:end_idx] = float(i_amp)
        result = NMModelHH().simulate(n_pts, 0.0, xdelta, i_ext)
        peaks, _ = find_peaks(result["V"], height=-20.0, distance=round(5.0 / xdelta))
        assert len(peaks) == expected_aps, (
            "i_amp=%d pA: expected %d AP(s), got %d" % (i_amp, expected_aps, len(peaks))
        )

    @pytest.mark.parametrize("i_amp,aps_during,aps_after", [
        (-30, 0, 0),
        (-35, 0, 1),
    ])
    def test_anodal_break(self, i_amp, aps_during, aps_after):
        """Hyperpolarising step: no spikes during step; -35 pA triggers one rebound AP."""
        from scipy.signal import find_peaks
        xdelta = 0.025
        onset_idx = round(25.0 / xdelta)
        end_idx   = onset_idx + round(100.0 / xdelta)
        n_pts     = round(175.0 / xdelta)   # 50 ms tail to catch the rebound
        i_ext = np.zeros(n_pts)
        i_ext[onset_idx:end_idx] = float(i_amp)
        result = NMModelHH().simulate(n_pts, 0.0, xdelta, i_ext)
        V = result["V"]
        min_dist = round(5.0 / xdelta)
        peaks, _ = find_peaks(V, height=-20.0, distance=min_dist)
        during = sum(1 for p in peaks if onset_idx <= p < end_idx)
        after  = sum(1 for p in peaks if p >= end_idx)
        assert during == aps_during, (
            "i_amp=%d pA: expected %d spike(s) during step, got %d" % (i_amp, aps_during, during)
        )
        assert after == aps_after, (
            "i_amp=%d pA: expected %d rebound spike(s) after step, got %d" % (i_amp, aps_after, after)
        )


# ──────────────────────────────────────────────────────────────────────────────
# Temperature (Q10) effect
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelHHTemperature:
    """Compare AP timing at two temperatures where the model reliably fires.

    Use tau_q10=1.5 so that the Q10 factor stays in a physiologically
    reasonable range at both test temperatures.
    """

    def _peak_time(self, temperature, tau_q10=1.5):
        m = NMModelHH()
        m.temperature = temperature
        m.tau_q10 = tau_q10
        i_ext = _make_i_ext(N_POINTS, I_ONSET_IDX, I_AMP_SUPRA)
        result = m.simulate(N_POINTS, XSTART, XDELTA, i_ext)
        V = result["V"]
        return int(np.argmax(V)) * XDELTA

    def test_higher_temperature_faster_ap(self):
        """At higher temperature, Q10 speeds up kinetics → earlier AP peak."""
        t_cold = self._peak_time(6.3)    # reference temperature: Q10 factor = 1.0
        t_warm = self._peak_time(16.3)   # 10°C above reference: Q10 factor = 1.5
        assert t_warm < t_cold


# ──────────────────────────────────────────────────────────────────────────────
# Derived quantities
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelHHDerived:
    def test_surface_area(self):
        m = NMModelHH()
        assert m._surface_area() == pytest.approx(1200.0)

    def test_capacitance(self):
        m = NMModelHH()
        assert m._capacitance() == pytest.approx(12.0)

    def test_q10_factor_at_ref_temp(self):
        m = NMModelHH()
        m.temperature = 6.3   # at reference → factor = 1.0
        assert m._q10_factor() == pytest.approx(1.0)

    def test_q10_factor_above_ref(self):
        m = NMModelHH()
        m.temperature = 16.3  # 10 °C above ref → factor = tau_q10^1
        assert m._q10_factor() == pytest.approx(m.tau_q10)


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelHHSerialisation:
    def test_to_dict_keys(self):
        m = NMModelHH()
        d = m.to_dict()
        assert d["model"] == "hh"
        for key in ("v0", "temperature", "cm_density", "diameter", "tau_q10"):
            assert key in d
        assert "conductances" in d

    def test_to_dict_conductances_list(self):
        m = NMModelHH()
        d = m.to_dict()
        assert isinstance(d["conductances"], list)
        assert len(d["conductances"]) == 3

    def test_from_dict_round_trip(self):
        m = NMModelHH()
        m.v0 = -70.0
        m.diameter = 25.0
        m2 = NMModelHH.from_dict(m.to_dict())
        assert m == m2

    def test_eq_symmetric(self):
        m1 = NMModelHH()
        m2 = NMModelHH()
        assert m1 == m2

    def test_eq_different_param(self):
        m1 = NMModelHH()
        m2 = NMModelHH()
        m2.diameter = 30.0
        assert m1 != m2

    def test_config_set_conductances(self):
        m = NMModelHH()
        m2 = NMModelHH()
        m2.conductances["Na"].g_density = 0.5
        d = m2.to_dict()
        m._config_set(d)
        assert m.conductances["Na"].g_density == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

class TestModelFactory:
    def test_dispatch_hh(self):
        m = NMModelHH()
        m2 = _model_from_dict(m.to_dict())
        assert isinstance(m2, NMModelHH)
        assert m == m2

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            _model_from_dict({"model": "mystery"})

    def test_none_model_raises(self):
        with pytest.raises(KeyError):
            _model_from_dict({})
