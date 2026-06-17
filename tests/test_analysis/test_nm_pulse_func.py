#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for nm_pulse_func: NMPulseFunc hierarchy and _pulse_func_from_dict."""
import math
import unittest

import numpy as np

from pyneuromatic.analysis.nm_pulse_func import (
    NMPulseFunc,
    NMPulseFuncSquare,
    NMPulseFuncRamp,
    NMPulseFuncExp,
    NMPulseFuncAlpha,
    NMPulseFuncSin,
    NMPulseFuncSinZap,
    NMPulseFuncUser,
    _pulse_func_from_dict,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _wave(func, n=100, xstart=0.0, xdelta=1.0, amp=1.0, onset=0.0, duration=math.inf):
    return func.waveform(n, xstart, xdelta, amp, onset, duration)


# ---------------------------------------------------------------------------
# Square
# ---------------------------------------------------------------------------

class TestNMPulseFuncSquare(unittest.TestCase):

    def setUp(self):
        self.f = NMPulseFuncSquare()

    def test_name(self):
        self.assertEqual(self.f.name, "square")

    def test_to_dict(self):
        self.assertEqual(self.f.to_dict(), {"pulse": "square"})

    def test_waveform_inside_window(self):
        y = _wave(self.f, amp=2.0, onset=10.0, duration=5.0)
        self.assertAlmostEqual(y[10], 2.0)
        self.assertAlmostEqual(y[14], 2.0)

    def test_waveform_outside_window(self):
        y = _wave(self.f, amp=2.0, onset=10.0, duration=5.0)
        self.assertAlmostEqual(y[9], 0.0)
        self.assertAlmostEqual(y[15], 0.0)

    def test_inf_duration_step(self):
        y = _wave(self.f, amp=1.0, onset=10.0)
        self.assertAlmostEqual(y[9], 0.0)
        self.assertAlmostEqual(y[10], 1.0)
        self.assertAlmostEqual(y[99], 1.0)

    def test_eq(self):
        self.assertEqual(self.f, NMPulseFuncSquare())

    def test_bad_name(self):
        with self.assertRaises(ValueError):
            NMPulseFuncSquare(name="exp")

    def test_no_extra_keys(self):
        with self.assertRaises(KeyError):
            _pulse_func_from_dict({"pulse": "square", "tau": 5.0})


# ---------------------------------------------------------------------------
# Ramp
# ---------------------------------------------------------------------------

class TestNMPulseFuncRamp(unittest.TestCase):

    def test_ramp_plus_rises(self):
        f = NMPulseFuncRamp("ramp+")
        y = _wave(f, n=11, amp=1.0, onset=0.0, duration=10.0)
        self.assertAlmostEqual(y[0], 0.0, places=5)
        self.assertAlmostEqual(y[9], 0.9, places=5)
        self.assertAlmostEqual(y[10], 0.0)

    def test_ramp_minus_falls(self):
        f = NMPulseFuncRamp("ramp-")
        y = _wave(f, n=11, amp=1.0, onset=0.0, duration=10.0)
        self.assertAlmostEqual(y[0], 1.0, places=5)
        self.assertAlmostEqual(y[9], 0.1, places=5)
        self.assertAlmostEqual(y[10], 0.0)

    def test_inf_duration_reaches_amp_at_end(self):
        f = NMPulseFuncRamp("ramp+")
        y = _wave(f, n=11, xdelta=1.0, amp=1.0, onset=0.0)
        self.assertAlmostEqual(y[0], 0.0, places=5)
        self.assertAlmostEqual(y[10], 1.0, places=5)

    def test_to_dict(self):
        self.assertEqual(NMPulseFuncRamp("ramp+").to_dict(), {"pulse": "ramp+"})
        self.assertEqual(NMPulseFuncRamp("ramp-").to_dict(), {"pulse": "ramp-"})

    def test_bad_name(self):
        with self.assertRaises(ValueError):
            NMPulseFuncRamp("ramp")


# ---------------------------------------------------------------------------
# Exp
# ---------------------------------------------------------------------------

class TestNMPulseFuncExp(unittest.TestCase):

    def setUp(self):
        self.f = NMPulseFuncExp(tau=10.0)

    def test_name(self):
        self.assertEqual(self.f.name, "exp")

    def test_tau_property(self):
        self.assertAlmostEqual(self.f.tau, 10.0)

    def test_to_dict(self):
        self.assertEqual(self.f.to_dict(), {"pulse": "exp", "tau": 10.0})

    def test_amp_at_onset(self):
        y = _wave(self.f, amp=3.0, onset=0.0)
        self.assertAlmostEqual(y[0], 3.0)

    def test_decays(self):
        y = _wave(self.f, amp=1.0, onset=0.0)
        self.assertGreater(y[0], y[10])

    def test_value_at_one_tau(self):
        # at x = onset + tau, y = amp * exp(-1)
        y = _wave(self.f, amp=1.0, onset=0.0)
        self.assertAlmostEqual(y[10], math.exp(-1.0), places=10)

    def test_bad_tau(self):
        with self.assertRaises(ValueError):
            NMPulseFuncExp(tau=0.0)
        with self.assertRaises(ValueError):
            NMPulseFuncExp(tau=-1.0)


# ---------------------------------------------------------------------------
# Alpha
# ---------------------------------------------------------------------------

class TestNMPulseFuncAlpha(unittest.TestCase):

    def setUp(self):
        self.f = NMPulseFuncAlpha(tau=10.0)

    def test_name(self):
        self.assertEqual(self.f.name, "alpha")

    def test_peak_at_onset_plus_tau(self):
        y = _wave(self.f, n=100, amp=1.0, onset=0.0)
        self.assertAlmostEqual(int(np.argmax(y)), 10, delta=1)

    def test_to_dict(self):
        self.assertEqual(self.f.to_dict(), {"pulse": "alpha", "tau": 10.0})


# ---------------------------------------------------------------------------
# Sin
# ---------------------------------------------------------------------------

class TestNMPulseFuncSin(unittest.TestCase):

    def setUp(self):
        self.f = NMPulseFuncSin(freq=0.1)   # period = 10 samples at xdelta=1

    def test_name(self):
        self.assertEqual(self.f.name, "sin")

    def test_zero_at_onset(self):
        y = _wave(self.f, amp=1.0, onset=0.0)
        self.assertAlmostEqual(y[0], 0.0, places=10)

    def test_peak_at_quarter_period(self):
        # freq=0.25 → period=4 samples; peak at sample 1 (t=1/4 period)
        f = NMPulseFuncSin(freq=0.25)
        y = _wave(f, n=20, amp=2.0, onset=0.0)
        self.assertAlmostEqual(y[1], 2.0, places=10)

    def test_phase_shift(self):
        f_cos = NMPulseFuncSin(freq=0.1, phase=math.pi / 2)
        y = _wave(f_cos, amp=1.0, onset=0.0)
        self.assertAlmostEqual(y[0], 1.0, places=4)   # sin(π/2) = 1

    def test_zero_before_onset(self):
        y = _wave(self.f, amp=1.0, onset=10.0)
        self.assertAlmostEqual(y[9], 0.0)

    def test_truncated_by_duration(self):
        y = _wave(self.f, amp=1.0, onset=0.0, duration=5.0)
        self.assertAlmostEqual(y[5], 0.0)

    def test_to_dict(self):
        self.assertEqual(self.f.to_dict(), {"pulse": "sin", "freq": 0.1, "phase": 0.0})

    def test_bad_freq(self):
        with self.assertRaises(ValueError):
            NMPulseFuncSin(freq=0.0)
        with self.assertRaises(ValueError):
            NMPulseFuncSin(freq=-1.0)

    def test_freq_setter(self):
        f = NMPulseFuncSin(freq=1.0)
        f.freq = 2.0
        self.assertAlmostEqual(f.freq, 2.0)

    def test_factory_roundtrip(self):
        d = self.f.to_dict()
        f2 = _pulse_func_from_dict(d)
        self.assertEqual(f2, self.f)


# ---------------------------------------------------------------------------
# SinZap
# ---------------------------------------------------------------------------

class TestNMPulseFuncSinZap(unittest.TestCase):

    def setUp(self):
        # 100 samples at xdelta=0.01 → 1 second total
        # f0=1 Hz, f1=10 Hz over 1 s
        self.f = NMPulseFuncSinZap(f0=1.0, f1=10.0)

    def test_name(self):
        self.assertEqual(self.f.name, "sinzap")

    def test_to_dict(self):
        self.assertEqual(self.f.to_dict(), {"pulse": "sinzap", "f0": 1.0, "f1": 10.0})

    def test_output_length(self):
        y = self.f.waveform(100, 0.0, 0.01, 1.0, 0.0, math.inf)
        self.assertEqual(len(y), 100)

    def test_amplitude_bounded(self):
        y = self.f.waveform(1000, 0.0, 0.001, 2.0, 0.0, math.inf)
        self.assertLessEqual(np.max(np.abs(y)), 2.0 + 1e-10)

    def test_zero_before_onset(self):
        y = self.f.waveform(100, 0.0, 0.01, 1.0, 0.1, math.inf)
        self.assertAlmostEqual(y[5], 0.0)   # x=0.05 < onset=0.1

    def test_bad_f0(self):
        with self.assertRaises(ValueError):
            NMPulseFuncSinZap(f0=0.0, f1=10.0)

    def test_bad_f1(self):
        with self.assertRaises(ValueError):
            NMPulseFuncSinZap(f0=1.0, f1=-1.0)

    def test_f0_setter(self):
        f = NMPulseFuncSinZap(f0=1.0, f1=10.0)
        f.f0 = 2.0
        self.assertAlmostEqual(f.f0, 2.0)

    def test_factory_roundtrip(self):
        f2 = _pulse_func_from_dict(self.f.to_dict())
        self.assertEqual(f2, self.f)


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------

class TestNMPulseFuncUser(unittest.TestCase):

    def setUp(self):
        # one full sine cycle, 50 samples, data_xdelta=1.0
        t = np.linspace(0, 2 * math.pi, 50, endpoint=False)
        self.raw = np.sin(t)
        self.f = NMPulseFuncUser(data=self.raw, data_xdelta=1.0)

    def test_name(self):
        self.assertEqual(self.f.name, "user")

    def test_peak_matches_amp(self):
        y = _wave(self.f, n=200, xdelta=1.0, amp=3.0, onset=0.0)
        self.assertAlmostEqual(np.max(y), 3.0, places=3)

    def test_zero_before_onset(self):
        y = _wave(self.f, n=100, xdelta=1.0, amp=1.0, onset=20.0)
        for i in range(20):
            self.assertAlmostEqual(y[i], 0.0)

    def test_truncated_by_duration(self):
        y = _wave(self.f, n=200, xdelta=1.0, amp=1.0, onset=0.0, duration=10.0)
        self.assertAlmostEqual(y[10], 0.0)
        self.assertAlmostEqual(y[50], 0.0)

    def test_resampled_to_target_xdelta(self):
        # target xdelta=0.5 → twice as many samples per source sample
        y = _wave(self.f, n=200, xdelta=0.5, amp=1.0, onset=0.0)
        self.assertAlmostEqual(np.max(y), 1.0, places=3)

    def test_to_dict_no_data(self):
        d = self.f.to_dict()
        self.assertEqual(d["pulse"], "user")
        self.assertAlmostEqual(d["data_xdelta"], 1.0)
        self.assertEqual(d["n_data"], 50)
        self.assertNotIn("data", d)

    def test_missing_data_raises(self):
        with self.assertRaises(KeyError):
            NMPulseFuncUser()

    def test_zero_data_returns_zeros(self):
        f = NMPulseFuncUser(data=np.zeros(10))
        y = _wave(f, n=50, amp=2.0, onset=0.0)
        np.testing.assert_array_equal(y, 0.0)

    def test_bad_data_xdelta(self):
        with self.assertRaises(ValueError):
            NMPulseFuncUser(data=self.raw, data_xdelta=0.0)

    def test_factory_requires_data(self):
        with self.assertRaises(KeyError):
            _pulse_func_from_dict({"pulse": "user"})


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestPulseFuncFactory(unittest.TestCase):

    def test_string_input(self):
        f = _pulse_func_from_dict("square")
        self.assertIsInstance(f, NMPulseFuncSquare)

    def test_all_shapes_by_name(self):
        shapes = ["square", "ramp+", "ramp-", "exp", "alpha", "sin", "sinzap"]
        for shape in shapes:
            f = _pulse_func_from_dict(shape)
            self.assertEqual(f.name, shape)

    def test_user_with_data(self):
        data = np.ones(10)
        f = _pulse_func_from_dict({"pulse": "user", "data": data})
        self.assertIsInstance(f, NMPulseFuncUser)

    def test_missing_pulse_key(self):
        with self.assertRaises(KeyError):
            _pulse_func_from_dict({"amp": 1.0})

    def test_unknown_shape(self):
        with self.assertRaises(ValueError):
            _pulse_func_from_dict("triangle")

    def test_unknown_key_for_square(self):
        with self.assertRaises(KeyError):
            _pulse_func_from_dict({"pulse": "square", "freq": 1.0})

    def test_exp_with_tau(self):
        f = _pulse_func_from_dict({"pulse": "exp", "tau": 5.0})
        self.assertIsInstance(f, NMPulseFuncExp)
        self.assertAlmostEqual(f.tau, 5.0)

    def test_sin_with_freq_phase(self):
        f = _pulse_func_from_dict({"pulse": "sin", "freq": 100.0, "phase": 0.5})
        self.assertAlmostEqual(f.freq, 100.0)
        self.assertAlmostEqual(f.phase, 0.5)

    def test_roundtrip_exp(self):
        f1 = NMPulseFuncExp(tau=7.5)
        f2 = _pulse_func_from_dict(f1.to_dict())
        self.assertEqual(f1, f2)

    def test_roundtrip_sinzap(self):
        f1 = NMPulseFuncSinZap(f0=2.0, f1=20.0)
        f2 = _pulse_func_from_dict(f1.to_dict())
        self.assertEqual(f1, f2)

    def test_not_dict_or_str_raises(self):
        with self.assertRaises(TypeError):
            _pulse_func_from_dict(42)


if __name__ == "__main__":
    unittest.main()
