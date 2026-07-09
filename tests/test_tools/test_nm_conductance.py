# -*- coding: utf-8 -*-
"""Tests for NMConductance class hierarchy."""
import math
import pytest

from pyneuromatic.tools.nm_conductance import (
    NMConductance,
    NMConductanceLeak,
    NMConductanceHHNa,
    NMConductanceHHK,
    NMConductanceGABA,
    NMConductanceAMPA,
    NMConductanceContainer,
    _conductance_from_dict,
)

V_REST = -65.0  # mV, resting potential used throughout


# ──────────────────────────────────────────────────────────────────────────────
# NMConductance base
# ──────────────────────────────────────────────────────────────────────────────

class TestNMConductanceBase:
    def test_invalid_name_empty(self):
        with pytest.raises(ValueError):
            NMConductance("", 0.003, -54.4)

    def test_invalid_g_density_negative(self):
        with pytest.raises(ValueError):
            NMConductanceLeak(g_density=-1.0)

    def test_invalid_g_density_bool(self):
        with pytest.raises(TypeError):
            NMConductanceLeak(g_density=True)

    def test_invalid_e_rev_bool(self):
        with pytest.raises(TypeError):
            NMConductanceLeak(e_rev=True)

    def test_g_density_setter_negative(self):
        c = NMConductanceLeak()
        with pytest.raises(ValueError):
            c.g_density = -0.1

    def test_g_density_setter_bool(self):
        c = NMConductanceLeak()
        with pytest.raises(TypeError):
            c.g_density = True

    def test_e_rev_setter_bool(self):
        c = NMConductanceLeak()
        with pytest.raises(TypeError):
            c.e_rev = False

    def test_eq_same(self):
        assert NMConductanceLeak() == NMConductanceLeak()

    def test_eq_different_g(self):
        a = NMConductanceLeak(g_density=0.003)
        b = NMConductanceLeak(g_density=0.005)
        assert a != b

    def test_repr(self):
        c = NMConductanceLeak()
        r = repr(c)
        assert "NMConductanceLeak" in r
        assert "g_density" in r


# ──────────────────────────────────────────────────────────────────────────────
# NMConductanceLeak
# ──────────────────────────────────────────────────────────────────────────────

class TestNMConductanceLeak:
    def test_defaults(self):
        c = NMConductanceLeak()
        assert c.g_density == pytest.approx(0.003)
        assert c.e_rev == pytest.approx(-54.387)
        assert c.name == "leak"

    def test_n_states(self):
        assert NMConductanceLeak().n_states() == 0

    def test_gate_names_empty(self):
        assert NMConductanceLeak().gate_names() == []

    def test_state_init_empty(self):
        assert NMConductanceLeak().state_init(V_REST) == []

    def test_current_at_rest(self):
        c = NMConductanceLeak()
        # I = g*(V - E_L) = 0.003 * (-65 - (-54.387))
        expected = 0.003 * (V_REST - (-54.387))
        assert c.current(V_REST, []) == pytest.approx(expected)

    def test_current_at_reversal(self):
        c = NMConductanceLeak()
        assert c.current(c.e_rev, []) == pytest.approx(0.0)

    def test_dydt_empty(self):
        assert NMConductanceLeak().dydt(V_REST, []) == []

    def test_dydt_scaled_empty(self):
        assert NMConductanceLeak().dydt_scaled(V_REST, [], q10=3.0) == []

    def test_to_dict(self):
        c = NMConductanceLeak(g_density=0.005, e_rev=-60.0)
        d = c.to_dict()
        assert d["conductance"] == "leak"
        assert d["g_density"] == pytest.approx(0.005)
        assert d["e_rev"] == pytest.approx(-60.0)

    def test_default_e_rev_matches_hh1952(self):
        """E_L default must be -54.387 mV (= -65 + 10.613, HH 1952 exact value)."""
        c = NMConductanceLeak()
        assert c.e_rev == pytest.approx(-54.387)

    def test_from_dict_round_trip(self):
        c = NMConductanceLeak(g_density=0.005, e_rev=-60.0)
        c2 = NMConductanceLeak.from_dict(c.to_dict())
        assert c2 == c

    def test_custom_params(self):
        c = NMConductanceLeak(g_density=0.01, e_rev=-70.0)
        assert c.g_density == pytest.approx(0.01)
        assert c.e_rev == pytest.approx(-70.0)


# ──────────────────────────────────────────────────────────────────────────────
# NMConductanceHHNa
# ──────────────────────────────────────────────────────────────────────────────

class TestNMConductanceHHNa:
    def test_defaults(self):
        c = NMConductanceHHNa()
        assert c.g_density == pytest.approx(1.2)
        assert c.e_rev == pytest.approx(50.0)
        assert c.name == "hhna"

    def test_n_states(self):
        assert NMConductanceHHNa().n_states() == 2

    def test_gate_names(self):
        assert NMConductanceHHNa().gate_names() == ["m", "h"]

    def test_state_init_in_range(self):
        c = NMConductanceHHNa()
        m_inf, h_inf = c.state_init(V_REST)
        assert 0.0 < m_inf < 1.0
        assert 0.0 < h_inf < 1.0

    def test_state_init_m_small_at_rest(self):
        # m_inf at rest should be small (Na activation is low at -65 mV)
        c = NMConductanceHHNa()
        m_inf, _ = c.state_init(V_REST)
        assert m_inf < 0.1

    def test_state_init_h_large_at_rest(self):
        # h_inf at rest should be near 1 (Na inactivation is mostly off at -65 mV)
        c = NMConductanceHHNa()
        _, h_inf = c.state_init(V_REST)
        assert h_inf > 0.5

    def test_current_formula(self):
        c = NMConductanceHHNa()
        m, h = 0.5, 0.6
        expected = 1.2 * (0.5 ** 3) * 0.6 * (V_REST - 50.0)
        assert c.current(V_REST, [m, h]) == pytest.approx(expected)

    def test_current_at_reversal(self):
        c = NMConductanceHHNa()
        assert c.current(c.e_rev, [0.5, 0.5]) == pytest.approx(0.0)

    def test_dydt_near_zero_at_steady_state(self):
        # At steady-state initial conditions, dm/dt and dh/dt should be near zero
        c = NMConductanceHHNa()
        m_inf, h_inf = c.state_init(V_REST)
        dm, dh = c.dydt(V_REST, [m_inf, h_inf])
        assert abs(dm) < 1e-10
        assert abs(dh) < 1e-10

    def test_alpha_m_singularity(self):
        # V = -40 mV: limit of alpha_m should be 1.0
        c = NMConductanceHHNa()
        assert c._alpha_m(-40.0) == pytest.approx(1.0)

    def test_alpha_m_non_singular(self):
        c = NMConductanceHHNa()
        alpha = c._alpha_m(-50.0)
        assert alpha > 0.0

    def test_beta_m_positive(self):
        c = NMConductanceHHNa()
        assert c._beta_m(V_REST) > 0.0

    def test_alpha_h_positive(self):
        c = NMConductanceHHNa()
        assert c._alpha_h(V_REST) > 0.0

    def test_beta_h_in_range(self):
        c = NMConductanceHHNa()
        bh = c._beta_h(V_REST)
        assert 0.0 < bh < 1.0

    def test_dydt_scaled(self):
        c = NMConductanceHHNa()
        m, h = 0.3, 0.7
        raw = c.dydt(V_REST, [m, h])
        scaled = c.dydt_scaled(V_REST, [m, h], q10=3.0)
        assert scaled[0] == pytest.approx(raw[0] * 3.0)
        assert scaled[1] == pytest.approx(raw[1] * 3.0)

    def test_to_dict(self):
        c = NMConductanceHHNa(g_density=1.0, e_rev=55.0)
        d = c.to_dict()
        assert d["conductance"] == "hhna"
        assert d["g_density"] == pytest.approx(1.0)
        assert d["e_rev"] == pytest.approx(55.0)

    def test_from_dict_round_trip(self):
        c = NMConductanceHHNa(g_density=1.0, e_rev=55.0)
        c2 = NMConductanceHHNa.from_dict(c.to_dict())
        assert c2 == c


# ──────────────────────────────────────────────────────────────────────────────
# NMConductanceHHK
# ──────────────────────────────────────────────────────────────────────────────

class TestNMConductanceHHK:
    def test_defaults(self):
        c = NMConductanceHHK()
        assert c.g_density == pytest.approx(0.36)
        assert c.e_rev == pytest.approx(-77.0)
        assert c.name == "hhk"

    def test_n_states(self):
        assert NMConductanceHHK().n_states() == 1

    def test_gate_names(self):
        assert NMConductanceHHK().gate_names() == ["n"]

    def test_state_init_in_range(self):
        c = NMConductanceHHK()
        (n_inf,) = c.state_init(V_REST)
        assert 0.0 < n_inf < 1.0

    def test_current_formula(self):
        c = NMConductanceHHK()
        n = 0.4
        expected = 0.36 * (n ** 4) * (V_REST - (-77.0))
        assert c.current(V_REST, [n]) == pytest.approx(expected)

    def test_current_at_reversal(self):
        c = NMConductanceHHK()
        assert c.current(c.e_rev, [0.5]) == pytest.approx(0.0)

    def test_dydt_near_zero_at_steady_state(self):
        c = NMConductanceHHK()
        (n_inf,) = c.state_init(V_REST)
        (dn,) = c.dydt(V_REST, [n_inf])
        assert abs(dn) < 1e-10

    def test_alpha_n_singularity(self):
        c = NMConductanceHHK()
        assert c._alpha_n(-55.0) == pytest.approx(0.1)

    def test_alpha_n_non_singular(self):
        c = NMConductanceHHK()
        assert c._alpha_n(-60.0) > 0.0

    def test_beta_n_positive(self):
        c = NMConductanceHHK()
        assert c._beta_n(V_REST) > 0.0

    def test_dydt_scaled(self):
        c = NMConductanceHHK()
        n = 0.3
        (raw_dn,) = c.dydt(V_REST, [n])
        (scaled_dn,) = c.dydt_scaled(V_REST, [n], q10=2.0)
        assert scaled_dn == pytest.approx(raw_dn * 2.0)

    def test_to_dict(self):
        c = NMConductanceHHK(g_density=0.4, e_rev=-80.0)
        d = c.to_dict()
        assert d["conductance"] == "hhk"
        assert d["g_density"] == pytest.approx(0.4)
        assert d["e_rev"] == pytest.approx(-80.0)

    def test_from_dict_round_trip(self):
        c = NMConductanceHHK(g_density=0.4, e_rev=-80.0)
        c2 = NMConductanceHHK.from_dict(c.to_dict())
        assert c2 == c


# ──────────────────────────────────────────────────────────────────────────────
# NMConductanceContainer
# ──────────────────────────────────────────────────────────────────────────────

class TestNMConductanceContainer:
    def _default_container(self):
        c = NMConductanceContainer()
        c.add("Leak", NMConductanceLeak())
        c.add("Na",   NMConductanceHHNa())
        c.add("K",    NMConductanceHHK())
        return c

    def test_add_and_len(self):
        c = self._default_container()
        assert len(c) == 3

    def test_contains(self):
        c = self._default_container()
        assert "Na" in c
        assert "X" not in c

    def test_getitem(self):
        c = self._default_container()
        assert isinstance(c["Leak"], NMConductanceLeak)

    def test_getitem_missing(self):
        c = NMConductanceContainer()
        with pytest.raises(KeyError):
            _ = c["missing"]

    def test_iter_yields_name_cond_pairs(self):
        c = self._default_container()
        pairs = list(c)
        assert len(pairs) == 3
        names = [name for name, _ in pairs]
        assert "Leak" in names and "Na" in names and "K" in names

    def test_add_invalid_name(self):
        c = NMConductanceContainer()
        with pytest.raises(ValueError):
            c.add("", NMConductanceLeak())

    def test_add_invalid_type(self):
        c = NMConductanceContainer()
        with pytest.raises(TypeError):
            c.add("X", "not a conductance")

    def test_to_dict(self):
        c = self._default_container()
        d = c.to_dict()
        assert "conductances" in d
        assert len(d["conductances"]) == 3
        names = [entry["name"] for entry in d["conductances"]]
        assert "Leak" in names

    def test_from_dict_round_trip(self):
        c = self._default_container()
        c2 = NMConductanceContainer.from_dict(c.to_dict())
        assert c2 == c

    def test_replace_existing_key(self):
        c = NMConductanceContainer()
        c.add("Na", NMConductanceHHNa(g_density=1.2))
        c.add("Na", NMConductanceHHNa(g_density=0.5))
        assert c["Na"].g_density == pytest.approx(0.5)
        assert len(c) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Synaptic conductances
# ──────────────────────────────────────────────────────────────────────────────

class TestNMConductanceGABA:
    def test_name(self):
        assert NMConductanceGABA().name == "gaba"

    def test_default_e_rev(self):
        assert NMConductanceGABA().e_rev == pytest.approx(-70.0)

    def test_default_g_density(self):
        assert NMConductanceGABA().g_density == pytest.approx(0.0)

    def test_current_is_zero(self):
        c = NMConductanceGABA()
        assert c.current(-70.0, []) == pytest.approx(0.0)
        assert c.current(-40.0, []) == pytest.approx(0.0)

    def test_e_rev_setter(self):
        c = NMConductanceGABA()
        c.e_rev = -75.0
        assert c.e_rev == pytest.approx(-75.0)

    def test_to_dict_round_trip(self):
        c = NMConductanceGABA(e_rev=-72.0)
        c2 = _conductance_from_dict(c.to_dict())
        assert c == c2


class TestNMConductanceAMPA:
    def test_name(self):
        assert NMConductanceAMPA().name == "ampa"

    def test_default_e_rev(self):
        assert NMConductanceAMPA().e_rev == pytest.approx(0.0)

    def test_default_g_density(self):
        assert NMConductanceAMPA().g_density == pytest.approx(0.0)

    def test_current_is_zero(self):
        c = NMConductanceAMPA()
        assert c.current(0.0, []) == pytest.approx(0.0)
        assert c.current(-60.0, []) == pytest.approx(0.0)

    def test_e_rev_setter(self):
        c = NMConductanceAMPA()
        c.e_rev = 5.0
        assert c.e_rev == pytest.approx(5.0)

    def test_to_dict_round_trip(self):
        c = NMConductanceAMPA(e_rev=2.0)
        c2 = _conductance_from_dict(c.to_dict())
        assert c == c2


class TestConductanceFactory:
    def test_dispatch_leak(self):
        d = {"conductance": "leak", "g_density": 0.003, "e_rev": -54.4}
        c = _conductance_from_dict(d)
        assert isinstance(c, NMConductanceLeak)

    def test_dispatch_hhna(self):
        d = {"conductance": "hhna", "g_density": 1.2, "e_rev": 50.0}
        c = _conductance_from_dict(d)
        assert isinstance(c, NMConductanceHHNa)

    def test_dispatch_hhk(self):
        d = {"conductance": "hhk", "g_density": 0.36, "e_rev": -77.0}
        c = _conductance_from_dict(d)
        assert isinstance(c, NMConductanceHHK)

    def test_dispatch_gaba(self):
        c = _conductance_from_dict({"conductance": "gaba", "e_rev": -70.0})
        assert isinstance(c, NMConductanceGABA)

    def test_dispatch_ampa(self):
        c = _conductance_from_dict({"conductance": "ampa", "e_rev": 0.0})
        assert isinstance(c, NMConductanceAMPA)

    def test_unknown_type_raises(self):
        with pytest.raises(KeyError):
            _conductance_from_dict({"conductance": "mystery"})

    def test_none_type_raises(self):
        with pytest.raises(KeyError):
            _conductance_from_dict({})

    def test_preserves_params(self):
        d = {"conductance": "leak", "g_density": 0.005, "e_rev": -60.0}
        c = _conductance_from_dict(d)
        assert c.g_density == pytest.approx(0.005)
        assert c.e_rev == pytest.approx(-60.0)
