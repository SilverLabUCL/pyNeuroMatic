# -*- coding: utf-8 -*-
"""Tests for NMModel and NMModelHH."""
import math
import numpy as np
import pytest

from pyneuromatic.tools.nm_model import NMModel, NMModelHH, NMModelIAF, _model_from_dict
from pyneuromatic.tools.nm_conductance import (
    NMConductanceLeak,
    NMConductanceHHNa,
    NMConductanceHHK,
    NMConductanceGABA,
    NMConductanceAMPA,
    NMConductanceNMDA,
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

    def test_dispatch_iaf(self):
        m = NMModelIAF()
        m2 = _model_from_dict(m.to_dict())
        assert isinstance(m2, NMModelIAF)
        assert m == m2

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            _model_from_dict({"model": "mystery"})

    def test_none_model_raises(self):
        with pytest.raises(KeyError):
            _model_from_dict({})


# ──────────────────────────────────────────────────────────────────────────────
# NMModelIAF — helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sim_default_iaf(i_amp=200.0):
    """Suprathreshold IAF simulation: 25 ms onset, 100 ms step, 150 ms total."""
    n_pts = round(150.0 / XDELTA)
    m = NMModelIAF()
    i_ext = _make_i_ext(n_pts, round(25.0 / XDELTA), i_amp,
                        duration_idx=round(100.0 / XDELTA))
    return m.simulate(n_pts, 0.0, XDELTA, i_ext)


# ──────────────────────────────────────────────────────────────────────────────
# NMModelIAF construction
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelIAFConstruct:
    def test_default_params(self):
        import math
        m = NMModelIAF()
        assert m.v0           == pytest.approx(-80.0)
        assert m.temperature  == pytest.approx(37.0)
        assert m.cm_density   == pytest.approx(0.01)
        assert m.diameter     == pytest.approx(10.0)
        assert m.ap_threshold == pytest.approx(-40.0)
        assert m.ap_peak      == pytest.approx(32.0)
        assert m.ap_reset     == pytest.approx(-61.0)
        assert m.ap_refrac    == pytest.approx(2.0)

    def test_default_conductances(self):
        m = NMModelIAF()
        assert "Leak" in m.conductances
        assert len(m.conductances) == 1

    def test_default_leak_params(self):
        import math
        m = NMModelIAF()
        leak = m.conductances["Leak"]
        expected_g = 0.9 / (math.pi * 10.0 ** 2)
        assert leak.g_density == pytest.approx(expected_g)
        assert leak.e_rev     == pytest.approx(-80.0)

    def test_name(self):
        m = NMModelIAF(name="test_iaf")
        assert m.name == "test_iaf"

    def test_v0_setter(self):
        m = NMModelIAF()
        m.v0 = -70.0
        assert m.v0 == pytest.approx(-70.0)

    def test_cm_density_setter_zero_raises(self):
        m = NMModelIAF()
        with pytest.raises(ValueError):
            m.cm_density = 0.0

    def test_diameter_setter_zero_raises(self):
        m = NMModelIAF()
        with pytest.raises(ValueError):
            m.diameter = 0.0

    def test_ap_refrac_setter_zero_raises(self):
        m = NMModelIAF()
        with pytest.raises(ValueError):
            m.ap_refrac = 0.0

    def test_ap_threshold_setter_bool_raises(self):
        m = NMModelIAF()
        with pytest.raises(TypeError):
            m.ap_threshold = True

    def test_ap_peak_setter(self):
        m = NMModelIAF()
        m.ap_peak = 40.0
        assert m.ap_peak == pytest.approx(40.0)

    def test_ap_reset_setter(self):
        m = NMModelIAF()
        m.ap_reset = -70.0
        assert m.ap_reset == pytest.approx(-70.0)

    def test_config_kwarg(self):
        m = NMModelIAF(config={"v0": -75.0, "ap_threshold": -35.0})
        assert m.v0           == pytest.approx(-75.0)
        assert m.ap_threshold == pytest.approx(-35.0)

    def test_config_unknown_key_raises(self):
        m = NMModelIAF()
        with pytest.raises(KeyError):
            m._config_set({"nonexistent_key": 1.0})

    def test_default_method_is_exact(self):
        assert NMModelIAF().method == "exact"

    def test_method_euler_accepted(self):
        m = NMModelIAF()
        m.method = "euler"
        assert m.method == "euler"

    def test_method_exact_accepted(self):
        m = NMModelIAF()
        m.method = "exact"
        assert m.method == "exact"

    def test_method_invalid_string_raises(self):
        m = NMModelIAF()
        with pytest.raises(ValueError):
            m.method = "rk45"

    def test_method_non_string_raises(self):
        m = NMModelIAF()
        with pytest.raises(TypeError):
            m.method = 1

    def test_config_kwarg_method(self):
        m = NMModelIAF(config={"method": "euler"})
        assert m.method == "euler"


# ──────────────────────────────────────────────────────────────────────────────
# simulate() return shape and keys
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelIAFSimulateShape:
    def test_returns_dict(self):
        assert isinstance(_sim_default_iaf(), dict)

    def test_has_V_key(self):
        assert "V" in _sim_default_iaf()

    def test_array_length(self):
        n_pts = round(150.0 / XDELTA)
        result = _sim_default_iaf()
        assert len(result["V"]) == n_pts

    def test_invalid_n_points(self):
        m = NMModelIAF()
        with pytest.raises(ValueError):
            m.simulate(0, 0.0, XDELTA, np.zeros(0))

    def test_invalid_xdelta(self):
        m = NMModelIAF()
        with pytest.raises(ValueError):
            m.simulate(100, 0.0, 0.0, np.zeros(100))

    def test_wrong_i_ext_length(self):
        m = NMModelIAF()
        with pytest.raises(ValueError):
            m.simulate(100, 0.0, XDELTA, np.zeros(50))


# ──────────────────────────────────────────────────────────────────────────────
# Resting state stability
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelIAFRest:
    def test_resting_potential_stable(self):
        """Zero input: Vm should stay within 1 mV of V0 (= E_leak)."""
        m = NMModelIAF()
        n_pts = round(100.0 / XDELTA)
        result = m.simulate(n_pts, 0.0, XDELTA, np.zeros(n_pts))
        assert np.all(np.abs(result["V"] - m.v0) < 1.0), (
            "Vm drifted > 1 mV from rest with no input"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Spike generation
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelIAFSpikes:
    def test_spike_peak_equals_ap_peak(self):
        """Peak of output voltage trace should equal ap_peak exactly."""
        m = NMModelIAF()
        result = _sim_default_iaf()
        assert result["V"].max() == pytest.approx(m.ap_peak)

    def test_subthreshold_no_spike(self):
        """30 pA (< threshold ≈ 36 pA) should produce no spike."""
        n_pts = round(150.0 / XDELTA)
        i_ext = _make_i_ext(n_pts, round(25.0 / XDELTA), amp=30.0,
                            duration_idx=round(100.0 / XDELTA))
        result = NMModelIAF().simulate(n_pts, 0.0, XDELTA, i_ext)
        assert result["V"].max() < 0.0, "Unexpected spike with subthreshold input"

    def test_refractory_enforces_minimum_isi(self):
        """All inter-spike intervals should be >= ap_refrac."""
        from scipy.signal import find_peaks
        n_pts = round(150.0 / XDELTA)
        i_ext = _make_i_ext(n_pts, round(25.0 / XDELTA), amp=500.0,
                            duration_idx=round(100.0 / XDELTA))
        m = NMModelIAF()
        result = m.simulate(n_pts, 0.0, XDELTA, i_ext)
        peaks, _ = find_peaks(result["V"], height=0.0, distance=round(1.5 / XDELTA))
        assert len(peaks) >= 2
        for p1, p2 in zip(peaks[:-1], peaks[1:]):
            isi_ms = (p2 - p1) * XDELTA
            assert isi_ms >= m.ap_refrac, (
                "ISI = %g ms < ap_refrac = %g ms" % (isi_ms, m.ap_refrac)
            )

    @pytest.mark.parametrize("i_amp,expected_aps", [
        (50,  20),
        (100, 34),
        (200, 42),
    ])
    def test_ap_count(self, i_amp, expected_aps):
        """Exact integration gives the analytically correct AP count.

        Onset=25 ms, duration=100 ms step.  Counts are determined by the
        closed-form ISI = τ_refrac + τ_m·ln((V_ss−V_reset)/(V_ss−V_thresh))
        and are stable at any step size with the exact method.
        """
        from scipy.signal import find_peaks
        n_pts = round(150.0 / XDELTA)
        i_ext = _make_i_ext(n_pts, round(25.0 / XDELTA), float(i_amp),
                            duration_idx=round(100.0 / XDELTA))
        m = NMModelIAF()
        m.method = "exact"
        result = m.simulate(n_pts, 0.0, XDELTA, i_ext)
        peaks, _ = find_peaks(result["V"], height=0.0, distance=round(1.5 / XDELTA))
        assert len(peaks) == expected_aps, (
            "%d pA: expected %d APs, got %d" % (i_amp, expected_aps, len(peaks))
        )

    def test_mean_isi(self):
        """At 50 pA the mean ISI should match the analytical prediction.

        Analytical: ISI = refrac + tau_m * ln((V_ss - V_reset)/(V_ss - V_thresh))
        With g_leak=0.9 nS, Cm=pi*10^2*0.01 pF, V_ss=-24.4 mV:  ISI ≈ 5.0 ms.
        """
        from scipy.signal import find_peaks
        n_pts = round(200.0 / XDELTA)
        i_ext = _make_i_ext(n_pts, round(25.0 / XDELTA), amp=50.0,
                            duration_idx=round(150.0 / XDELTA))
        result = NMModelIAF().simulate(n_pts, 0.0, XDELTA, i_ext)
        peaks, _ = find_peaks(result["V"], height=0.0, distance=round(1.5 / XDELTA))
        assert len(peaks) >= 5, "Too few APs to measure ISI"
        isis = [(peaks[k + 1] - peaks[k]) * XDELTA for k in range(1, len(peaks) - 1)]
        mean_isi = sum(isis) / len(isis)
        assert mean_isi == pytest.approx(5.0, abs=0.15), (
            "Mean ISI = %g ms, expected ≈ 5.0 ms" % mean_isi
        )

    @pytest.mark.parametrize("i_amp_lo,i_amp_hi", [
        (50, 100),
        (100, 200),
    ])
    def test_higher_current_shorter_isi(self, i_amp_lo, i_amp_hi):
        """More current should give a shorter (faster) mean ISI."""
        from scipy.signal import find_peaks
        n_pts = round(200.0 / XDELTA)
        dur_idx = round(150.0 / XDELTA)
        onset = round(25.0 / XDELTA)
        min_dist = round(1.5 / XDELTA)

        def mean_isi(amp):
            i_ext = _make_i_ext(n_pts, onset, amp=float(amp), duration_idx=dur_idx)
            result = NMModelIAF().simulate(n_pts, 0.0, XDELTA, i_ext)
            peaks, _ = find_peaks(result["V"], height=0.0, distance=min_dist)
            isis = [(peaks[k + 1] - peaks[k]) * XDELTA for k in range(1, len(peaks) - 1)]
            return sum(isis) / len(isis)

        assert mean_isi(i_amp_lo) > mean_isi(i_amp_hi), (
            "%d pA should have longer ISI than %d pA" % (i_amp_lo, i_amp_hi)
        )

    def test_exact_and_euler_agree_at_small_dt(self):
        """At dt=0.025 ms, exact and Euler give nearly identical subthreshold traces.

        Uses a strictly subthreshold current (< threshold I) so there are no
        spikes and no spike-timing offset artefacts between the two methods.
        """
        n_pts = round(150.0 / XDELTA)
        # 20 pA is well below threshold (I_thresh = g_leak * (V_thresh - E_leak) ≈ 36 pA)
        i_ext = _make_i_ext(n_pts, round(25.0 / XDELTA), amp=20.0,
                            duration_idx=round(100.0 / XDELTA))
        m_exact = NMModelIAF()
        m_exact.method = "exact"
        m_euler = NMModelIAF()
        m_euler.method = "euler"
        V_exact = m_exact.simulate(n_pts, 0.0, XDELTA, i_ext)["V"]
        V_euler = m_euler.simulate(n_pts, 0.0, XDELTA, i_ext)["V"]
        assert np.max(np.abs(V_exact - V_euler)) < 0.05, (
            "Exact and Euler disagree by more than 0.05 mV at dt=%g ms" % XDELTA
        )

    def test_exact_correct_at_large_dt(self):
        """Exact integration recovers the correct τ_m even at dt=0.5 ms where Euler errors."""
        import math
        xdelta_large = 0.5  # 20× default step — Euler accumulates large error
        n_pts = round(50.0 / xdelta_large)
        onset = round(5.0 / xdelta_large)
        i_amp = 10.0  # subthreshold

        m = NMModelIAF()
        SA = math.pi * m.diameter ** 2
        g_total = m.conductances["Leak"].g_density * SA
        Cm = m.cm_density * SA
        tau_m = Cm / g_total
        e_rev = m.conductances["Leak"].e_rev
        V_ss = e_rev + i_amp / g_total

        i_ext = _make_i_ext(n_pts, onset, amp=i_amp,
                            duration_idx=n_pts - onset)
        m.method = "exact"
        V_exact = m.simulate(n_pts, 0.0, xdelta_large, i_ext)["V"]

        # Compare against the analytical solution at each sample
        V_ana = np.array([
            V_ss + (m.v0 - V_ss) * math.exp(-(i - onset) * xdelta_large / tau_m)
            if i >= onset else m.v0
            for i in range(n_pts)
        ])
        assert np.max(np.abs(V_exact[onset:] - V_ana[onset:])) < 0.01, (
            "Exact integration deviates from analytical solution by more than 0.01 mV"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Derived quantities
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelIAFTimeConstant:
    def test_membrane_time_constant(self):
        """τ_m = Cm / g_leak measured from subthreshold exponential charging.

        Apply a step current, find the time at which V reaches 63.2% of the
        way from V0 to V_ss — that time equals τ_m.  Tolerance is ±2 × xdelta
        (one grid step either side of the true crossing).
        """
        xdelta = 0.025
        onset_idx = round(5.0 / xdelta)
        n_pts = round(50.0 / xdelta)   # 50 ms — much longer than τ_m ≈ 3.5 ms
        i_amp = 20.0                    # pA — subthreshold (threshold ≈ 36 pA)

        m = NMModelIAF()
        i_ext = np.zeros(n_pts)
        i_ext[onset_idx:] = i_amp
        V = m.simulate(n_pts, 0.0, xdelta, i_ext)["V"]

        # Expected: τ_m = Cm / g_leak  (SA cancels, so = cm_density / g_leak_density)
        g_leak = m.conductances["Leak"].g_density * m._surface_area()  # nS
        tau_expected = m._capacitance() / g_leak                        # ms

        # V_ss = E_leak + I/g_leak;  target = V0 + (V_ss - V0)*(1 - 1/e)
        import math
        V_ss = m.conductances["Leak"].e_rev + i_amp / g_leak
        V_target = m.v0 + (V_ss - m.v0) * (1.0 - 1.0 / math.e)

        crossing = np.argmax(V[onset_idx:] >= V_target)
        tau_measured = crossing * xdelta   # ms after onset

        assert tau_measured == pytest.approx(tau_expected, abs=2 * xdelta), (
            "τ_m measured=%g ms, expected=%g ms" % (tau_measured, tau_expected)
        )


class TestNMModelIAFDerived:
    def test_surface_area(self):
        import math
        m = NMModelIAF()
        assert m._surface_area() == pytest.approx(math.pi * 10.0 ** 2)

    def test_capacitance(self):
        import math
        m = NMModelIAF()
        assert m._capacitance() == pytest.approx(0.01 * math.pi * 10.0 ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation
# ──────────────────────────────────────────────────────────────────────────────

class TestNMModelIAFSerialisation:
    def test_to_dict_keys(self):
        m = NMModelIAF()
        d = m.to_dict()
        assert d["model"] == "iaf"
        for key in ("v0", "temperature", "cm_density", "diameter",
                    "ap_threshold", "ap_peak", "ap_reset", "ap_refrac", "method"):
            assert key in d
        assert "conductances" in d

    def test_to_dict_method_default(self):
        assert NMModelIAF().to_dict()["method"] == "exact"

    def test_method_round_trips_euler(self):
        m = NMModelIAF()
        m.method = "euler"
        m2 = NMModelIAF.from_dict(m.to_dict())
        assert m2.method == "euler"

    def test_to_dict_conductances_list(self):
        m = NMModelIAF()
        d = m.to_dict()
        assert isinstance(d["conductances"], list)
        assert len(d["conductances"]) == 1

    def test_from_dict_round_trip(self):
        m = NMModelIAF()
        m.ap_refrac = 3.0
        m.diameter  = 15.0
        m2 = NMModelIAF.from_dict(m.to_dict())
        assert m == m2

    def test_eq_symmetric(self):
        assert NMModelIAF() == NMModelIAF()

    def test_eq_different_param(self):
        m1 = NMModelIAF()
        m2 = NMModelIAF()
        m2.ap_threshold = -35.0
        assert m1 != m2


# ──────────────────────────────────────────────────────────────────────────────
# Synaptic conductances via g_ext
# ──────────────────────────────────────────────────────────────────────────────

def _iaf_with_syn():
    """Return an NMModelIAF with GABA and AMPA registered (method='exact')."""
    m = NMModelIAF()
    m.conductances.add("GABA", NMConductanceGABA())
    m.conductances.add("AMPA", NMConductanceAMPA())
    return m


class TestNMModelIAFSynaptic:
    def test_g_ext_none_unchanged(self):
        """simulate(g_ext=None) must give the same result as simulate() without g_ext."""
        n_pts = round(150.0 / XDELTA)
        i_ext = _make_i_ext(n_pts, round(25.0 / XDELTA), amp=200.0,
                            duration_idx=round(100.0 / XDELTA))
        m = _iaf_with_syn()
        V_base = m.simulate(n_pts, 0.0, XDELTA, i_ext)["V"]
        V_none = m.simulate(n_pts, 0.0, XDELTA, i_ext, g_ext=None)["V"]
        assert np.array_equal(V_base, V_none)

    def test_gaba_inhibits_firing(self):
        """Strong GABA conductance should reduce AP count vs no-GABA baseline."""
        from scipy.signal import find_peaks
        n_pts = round(150.0 / XDELTA)
        onset = round(25.0 / XDELTA)
        dur   = round(100.0 / XDELTA)
        i_ext = _make_i_ext(n_pts, onset, amp=200.0, duration_idx=dur)

        # Baseline: no GABA
        m_base = NMModelIAF()
        V_base = m_base.simulate(n_pts, 0.0, XDELTA, i_ext)["V"]
        peaks_base, _ = find_peaks(V_base, height=0.0, distance=round(1.5 / XDELTA))

        # With large GABA conductance step (1.0 nS, E_GABA=-70 mV pulls Vm down)
        m_gaba = _iaf_with_syn()
        g_gaba = _make_i_ext(n_pts, onset, amp=1.0, duration_idx=dur)
        V_gaba = m_gaba.simulate(n_pts, 0.0, XDELTA, i_ext, g_ext={"GABA": g_gaba})["V"]
        peaks_gaba, _ = find_peaks(V_gaba, height=0.0, distance=round(1.5 / XDELTA))

        assert len(peaks_gaba) < len(peaks_base), (
            "GABA should reduce AP count: baseline=%d, with GABA=%d"
            % (len(peaks_base), len(peaks_gaba))
        )

    def test_ampa_drives_subthreshold_to_fire(self):
        """Subthreshold i_ext alone gives no spikes; adding AMPA step produces spikes."""
        from scipy.signal import find_peaks
        n_pts = round(150.0 / XDELTA)
        onset = round(25.0 / XDELTA)
        dur   = round(100.0 / XDELTA)
        # 20 pA is subthreshold (threshold ≈ 36 pA)
        i_ext = _make_i_ext(n_pts, onset, amp=20.0, duration_idx=dur)

        # No AMPA: no spikes
        m = NMModelIAF()
        V_sub = m.simulate(n_pts, 0.0, XDELTA, i_ext)["V"]
        assert V_sub.max() < 0.0, "Expected no spike with 20 pA"

        # With AMPA conductance (0.5 nS, E_AMPA=0 mV pushes Vm up)
        m_syn = _iaf_with_syn()
        g_ampa = _make_i_ext(n_pts, onset, amp=0.5, duration_idx=dur)
        V_syn = m_syn.simulate(n_pts, 0.0, XDELTA, i_ext, g_ext={"AMPA": g_ampa})["V"]
        peaks, _ = find_peaks(V_syn, height=0.0, distance=round(1.5 / XDELTA))
        assert len(peaks) > 0, "Expected spikes with AMPA conductance"

    def test_g_ext_wrong_length_raises(self):
        m = _iaf_with_syn()
        n_pts = round(100.0 / XDELTA)
        i_ext = np.zeros(n_pts)
        with pytest.raises(ValueError):
            m.simulate(n_pts, 0.0, XDELTA, i_ext, g_ext={"GABA": np.zeros(n_pts + 10)})

    def test_g_ext_unknown_key_raises(self):
        m = NMModelIAF()  # no GABA/AMPA registered
        n_pts = round(100.0 / XDELTA)
        i_ext = np.zeros(n_pts)
        with pytest.raises(KeyError):
            m.simulate(n_pts, 0.0, XDELTA, i_ext, g_ext={"GABA": np.zeros(n_pts)})

    @pytest.mark.parametrize("g_gaba_nS,e_gaba", [
        (0.45, -70.0),   # half leak conductance — V_ss closer to E_leak
        (0.9,  -70.0),   # equal to leak — V_ss midway between E_leak and E_GABA
        (1.8,  -70.0),   # double leak — V_ss closer to E_GABA
        (0.9,  -65.0),   # different E_GABA (more depolarised)
    ])
    def test_gaba_steady_state_voltage(self, g_gaba_nS, e_gaba):
        """Constant GABA conductance drives Vm to a weighted mean of E_leak and E_GABA.

        Analytical steady state:
            V_ss = (G_leak * E_leak + G_GABA * E_GABA) / (G_leak + G_GABA)

        Run for 100 ms with no i_ext; that is >> any τ_m so the transient has
        decayed to < 0.001 mV.  Final Vm must equal V_ss within 0.01 mV.
        """
        n_pts = round(100.0 / XDELTA)
        m = NMModelIAF()
        m.conductances.add("GABA", NMConductanceGABA(e_rev=e_gaba))

        SA = math.pi * m.diameter ** 2
        G_leak = m.conductances["Leak"].g_density * SA
        E_leak = m.conductances["Leak"].e_rev
        V_ss_expected = (G_leak * E_leak + g_gaba_nS * e_gaba) / (G_leak + g_gaba_nS)

        g_gaba = np.full(n_pts, g_gaba_nS)
        V = m.simulate(n_pts, 0.0, XDELTA, np.zeros(n_pts),
                       g_ext={"GABA": g_gaba})["V"]

        # V_ss must lie strictly between the two reversal potentials
        assert min(E_leak, e_gaba) < V_ss_expected < max(E_leak, e_gaba)
        # Final Vm must match the analytical prediction
        assert V[-1] == pytest.approx(V_ss_expected, abs=0.01), (
            "g_GABA=%g nS, E_GABA=%g mV: expected V_ss=%g mV, got %g mV"
            % (g_gaba_nS, e_gaba, V_ss_expected, V[-1])
        )

    @pytest.mark.parametrize("g_ampa_nS,e_ampa", [
        (0.45, 0.0),    # half leak conductance — V_ss closer to E_leak
        (0.9,  0.0),    # equal to leak — V_ss midway between E_leak and E_AMPA
        (1.8,  0.0),    # double leak — V_ss closer to E_AMPA
        (0.9,  5.0),    # slightly different E_AMPA
    ])
    def test_ampa_steady_state_voltage(self, g_ampa_nS, e_ampa):
        """Constant AMPA conductance drives Vm to a weighted mean of E_leak and E_AMPA.

        AP threshold is raised to 200 mV so the neuron never fires and the
        membrane charges to the analytical steady state:
            V_ss = (G_leak * E_leak + G_AMPA * E_AMPA) / (G_leak + G_AMPA)
        """
        n_pts = round(100.0 / XDELTA)
        m = NMModelIAF()
        m.ap_threshold = 200.0   # prevent spiking — AMPA pushes Vm above default threshold
        m.conductances.add("AMPA", NMConductanceAMPA(e_rev=e_ampa))

        SA = math.pi * m.diameter ** 2
        G_leak = m.conductances["Leak"].g_density * SA
        E_leak = m.conductances["Leak"].e_rev
        V_ss_expected = (G_leak * E_leak + g_ampa_nS * e_ampa) / (G_leak + g_ampa_nS)

        g_ampa = np.full(n_pts, g_ampa_nS)
        V = m.simulate(n_pts, 0.0, XDELTA, np.zeros(n_pts),
                       g_ext={"AMPA": g_ampa})["V"]

        # V_ss must lie strictly between the two reversal potentials
        assert min(E_leak, e_ampa) < V_ss_expected < max(E_leak, e_ampa)
        # Final Vm must match the analytical prediction
        assert V[-1] == pytest.approx(V_ss_expected, abs=0.01), (
            "g_AMPA=%g nS, E_AMPA=%g mV: expected V_ss=%g mV, got %g mV"
            % (g_ampa_nS, e_ampa, V_ss_expected, V[-1])
        )

    def test_euler_and_exact_agree_with_g_ext(self):
        """Exact and Euler give nearly identical Vm at dt=0.025 ms with g_ext present."""
        n_pts = round(150.0 / XDELTA)
        onset = round(25.0 / XDELTA)
        dur   = round(100.0 / XDELTA)
        # Subthreshold to avoid spike-timing jitter between methods
        i_ext = _make_i_ext(n_pts, onset, amp=20.0, duration_idx=dur)
        g_ampa = _make_i_ext(n_pts, onset, amp=0.1, duration_idx=dur)

        m_exact = _iaf_with_syn()
        m_exact.method = "exact"
        m_euler = _iaf_with_syn()
        m_euler.method = "euler"

        V_exact = m_exact.simulate(n_pts, 0.0, XDELTA, i_ext, g_ext={"AMPA": g_ampa})["V"]
        V_euler = m_euler.simulate(n_pts, 0.0, XDELTA, i_ext, g_ext={"AMPA": g_ampa})["V"]
        assert np.max(np.abs(V_exact - V_euler)) < 0.05, (
            "Exact and Euler with g_ext disagree by > 0.05 mV at dt=%g ms" % XDELTA
        )

    def test_nmda_blocked_at_rest(self):
        """NMDA at resting Vm (−80 mV) should be strongly blocked: far less current than AMPA.

        The Boltzmann factor at −80 mV (v_half=−12.8, v_slope=22.4) is ≈ 0.03,
        so 1 nS NMDA produces only ~3% of the current a 1 nS AMPA step would.
        """
        n_pts = round(100.0 / XDELTA)
        g_1nS = np.ones(n_pts)  # constant 1 nS

        # AMPA (no block): 1 nS at E_AMPA=0 mV drives Vm well above rest
        m_ampa = NMModelIAF()
        m_ampa.ap_threshold = 200.0
        m_ampa.conductances.add("AMPA", NMConductanceAMPA())
        V_ampa = m_ampa.simulate(n_pts, 0.0, XDELTA, np.zeros(n_pts),
                                 g_ext={"AMPA": g_1nS})["V"]

        # NMDA (default Boltzmann block): 1 nS at E_NMDA=0 mV, same e_rev as AMPA
        m_nmda = NMModelIAF()
        m_nmda.ap_threshold = 200.0
        m_nmda.conductances.add("NMDA", NMConductanceNMDA())  # default boltzmann, v_half=-12.8
        V_nmda = m_nmda.simulate(n_pts, 0.0, XDELTA, np.zeros(n_pts),
                                 g_ext={"NMDA": g_1nS})["V"]

        # NMDA-driven depolarisation must be much smaller than AMPA-driven
        delta_ampa = V_ampa[-1] - m_ampa.v0
        delta_nmda = V_nmda[-1] - m_nmda.v0
        assert delta_nmda < delta_ampa * 0.2, (
            "NMDA at rest should be >80%% blocked: ΔAMPA=%g mV, ΔNMDA=%g mV"
            % (delta_ampa, delta_nmda)
        )

    def test_nmda_steady_state_with_full_unblock(self):
        """NMDA with v_half=-200 (B≈1 everywhere) matches the AMPA steady-state formula.

        V_ss = (G_leak * E_leak + G_NMDA * E_NMDA) / (G_leak + G_NMDA), within 0.01 mV.
        """
        g_nmda_nS = 0.9
        n_pts = round(100.0 / XDELTA)

        m = NMModelIAF()
        m.ap_threshold = 200.0
        m.conductances.add("NMDA", NMConductanceNMDA(
            mg_block="boltzmann", v_half=-500.0, v_slope=22.4  # B≈1 to < 1 ppb at physiological V
        ))

        SA = math.pi * m.diameter ** 2
        G_leak = m.conductances["Leak"].g_density * SA
        E_leak = m.conductances["Leak"].e_rev
        E_nmda = m.conductances["NMDA"].e_rev
        V_ss_expected = (G_leak * E_leak + g_nmda_nS * E_nmda) / (G_leak + g_nmda_nS)

        g_arr = np.full(n_pts, g_nmda_nS)
        V = m.simulate(n_pts, 0.0, XDELTA, np.zeros(n_pts), g_ext={"NMDA": g_arr})["V"]

        assert V[-1] == pytest.approx(V_ss_expected, abs=0.01), (
            "NMDA full-unblock V_ss: expected %g mV, got %g mV" % (V_ss_expected, V[-1])
        )

    def test_nmda_all_block_models_in_range(self):
        """voltage_factor(v) must be in (0, 1] for all block models at typical voltages."""
        for model in ("boltzmann", "jahr_stevens_1990", "gc_schwartz_2012"):
            c = NMConductanceNMDA(mg_block=model)
            for v in (-80.0, -40.0, 0.0, 50.0):
                b = c.voltage_factor(v)
                assert 0.0 < b <= 1.0, (
                    "model=%r, V=%g mV: voltage_factor=%g not in (0, 1]" % (model, v, b)
                )
