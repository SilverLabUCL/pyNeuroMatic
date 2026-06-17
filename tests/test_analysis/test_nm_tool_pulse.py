#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for nm_pulse and nm_tool_pulse: NMPulse, NMPulseContainer,
NMToolPulse, NMToolPulseConfig.
"""
import math
import unittest

import numpy as np

from pyneuromatic.analysis.nm_pulse import NMPulse, NMPulseContainer
from pyneuromatic.analysis.nm_tool_pulse import NMToolPulse, NMToolPulseConfig
from pyneuromatic.core.nm_data import NMData
from pyneuromatic.core.nm_folder import NMFolder
from pyneuromatic.core.nm_manager import NMManager, HIERARCHY_SELECT_KEYS

NM = NMManager(quiet=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N = 100
_XSTART = 0.0
_XDELTA = 1.0


def _make_empty_data(name, n=_N, xstart=_XSTART, xdelta=_XDELTA):
    return NMData(NM, name=name, nparray=np.zeros(n),
                  xscale={"start": xstart, "delta": xdelta})


def _make_targets(data_list, folder=None):
    if folder is None:
        folder = NMFolder(NM, name="TestFolder")
    return [
        {k: None for k in HIERARCHY_SELECT_KEYS} | {"folder": folder, "data": d}
        for d in data_list
    ], folder


def _run_tool(tool, data_list, folder=None):
    """Drive tool directly with a list of NMData."""
    targets, folder = _make_targets(data_list, folder=folder)
    tool.run_all(targets)
    return folder


# ---------------------------------------------------------------------------
# NMPulse — defaults
# ---------------------------------------------------------------------------

class TestNMPulseDefaults(unittest.TestCase):

    def setUp(self):
        self.p = NMPulse()

    def test_name(self):
        self.assertEqual(self.p.name, "NMPulse0")

    def test_enabled(self):
        self.assertTrue(self.p.enabled)

    def test_pulse(self):
        self.assertEqual(self.p.pulse, "square")

    def test_epoch(self):
        self.assertEqual(self.p.epoch, 0)

    def test_epoch_delta(self):
        self.assertEqual(self.p.epoch_delta, 0)

    def test_amp(self):
        self.assertEqual(self.p.amp, 1.0)

    def test_onset(self):
        self.assertEqual(self.p.onset, 10.0)

    def test_duration(self):
        self.assertTrue(math.isinf(self.p.duration))

    def test_amp_delta(self):
        self.assertEqual(self.p.amp_delta, 0.0)

    def test_onset_delta(self):
        self.assertEqual(self.p.onset_delta, 0.0)

    def test_duration_delta(self):
        self.assertEqual(self.p.duration_delta, 0.0)

    def test_amp_stdv(self):
        self.assertEqual(self.p.amp_stdv, 0.0)

    def test_onset_stdv(self):
        self.assertEqual(self.p.onset_stdv, 0.0)

    def test_duration_stdv(self):
        self.assertEqual(self.p.duration_stdv, 0.0)


# ---------------------------------------------------------------------------
# NMPulse — property validation
# ---------------------------------------------------------------------------

class TestNMPulseProperties(unittest.TestCase):

    def setUp(self):
        self.p = NMPulse()

    def test_pulse_valid_shapes(self):
        for shape in ("square", "ramp+", "ramp-", "exp", "alpha"):
            self.p.pulse = shape
            self.assertEqual(self.p.pulse, shape)

    def test_pulse_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.p.pulse = "triangle"

    def test_pulse_type_error(self):
        with self.assertRaises(TypeError):
            self.p.pulse = 42

    def test_epoch_int(self):
        self.p.epoch = 5
        self.assertEqual(self.p.epoch, 5)

    def test_epoch_all_shorthand(self):
        self.p.epoch = "all"
        self.assertEqual(self.p.epoch, 0)
        self.assertEqual(self.p.epoch_delta, 1)

    def test_epoch_negative_raises(self):
        with self.assertRaises(ValueError):
            self.p.epoch = -1

    def test_epoch_invalid_string_raises(self):
        with self.assertRaises(ValueError):
            self.p.epoch = "none"

    def test_epoch_bool_raises(self):
        with self.assertRaises(TypeError):
            self.p.epoch = True

    def test_epoch_delta_valid(self):
        self.p.epoch_delta = 3
        self.assertEqual(self.p.epoch_delta, 3)

    def test_epoch_delta_zero_allowed(self):
        self.p.epoch_delta = 0
        self.assertEqual(self.p.epoch_delta, 0)

    def test_epoch_delta_bool_raises(self):
        with self.assertRaises(TypeError):
            self.p.epoch_delta = True

    def test_float_attrs_accept_int(self):
        self.p.amp = 2
        self.assertAlmostEqual(self.p.amp, 2.0)

    def test_float_attrs_bool_raises(self):
        with self.assertRaises(TypeError):
            self.p.amp = True

    def test_float_attrs_string_raises(self):
        with self.assertRaises(TypeError):
            self.p.onset = "10"

    def test_enabled_toggle(self):
        self.p.enabled = False
        self.assertFalse(self.p.enabled)
        self.p.enabled = True
        self.assertTrue(self.p.enabled)

    def test_enabled_non_bool_raises(self):
        with self.assertRaises(TypeError):
            self.p.enabled = 1


# ---------------------------------------------------------------------------
# NMPulse — config dict
# ---------------------------------------------------------------------------

class TestNMPulseConfigSet(unittest.TestCase):

    def test_config_via_constructor(self):
        p = NMPulse(config={"pulse": "exp", "amp": 2.0, "onset": 5.0, "tau": 20.0})
        self.assertEqual(p.pulse, "exp")
        self.assertAlmostEqual(p.amp, 2.0)
        self.assertAlmostEqual(p.onset, 5.0)
        self.assertAlmostEqual(p.func.tau, 20.0)

    def test_to_dict_round_trip(self):
        p = NMPulse(config={"pulse": "ramp+", "epoch": 3, "epoch_delta": 2,
                             "amp": 4.0, "onset": 1.0, "duration": 5.0,
                             "onset_delta": 0.5})
        d = p.to_dict()
        p2 = NMPulse.from_dict(d)
        self.assertEqual(p, p2)

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            NMPulse(config={"bad_key": 1})

    def test_name_key_ignored(self):
        p = NMPulse(name="myp", config={"name": "ignored", "amp": 3.0})
        self.assertEqual(p.name, "myp")
        self.assertAlmostEqual(p.amp, 3.0)


# ---------------------------------------------------------------------------
# NMPulse — epoch targeting
# ---------------------------------------------------------------------------

class TestNMPulseEpochTargeting(unittest.TestCase):

    def test_default_fires_only_epoch_zero(self):
        p = NMPulse()  # epoch=0, epoch_delta=0
        self.assertTrue(p.targets_epoch(0))
        for i in range(1, 5):
            self.assertFalse(p.targets_epoch(i))

    def test_all_shorthand_fires_every_epoch(self):
        p = NMPulse(config={"epoch": "all"})  # → epoch=0, epoch_delta=1
        self.assertEqual(p.epoch, 0)
        self.assertEqual(p.epoch_delta, 1)
        for i in range(5):
            self.assertTrue(p.targets_epoch(i))

    def test_int_epoch_fires_only_on_that_epoch(self):
        # epoch=2, epoch_delta=0 (default) → fires only on epoch 2
        p = NMPulse(config={"epoch": 2})
        self.assertFalse(p.targets_epoch(0))
        self.assertFalse(p.targets_epoch(1))
        self.assertTrue(p.targets_epoch(2))
        self.assertFalse(p.targets_epoch(3))
        self.assertFalse(p.targets_epoch(4))

    def test_int_epoch_skips_before_start(self):
        p = NMPulse(config={"epoch": 5, "epoch_delta": 1})
        for i in range(5):
            self.assertFalse(p.targets_epoch(i))
        self.assertTrue(p.targets_epoch(5))

    def test_epoch_delta_stride(self):
        p = NMPulse(config={"epoch": 0, "epoch_delta": 2})
        self.assertTrue(p.targets_epoch(0))
        self.assertFalse(p.targets_epoch(1))
        self.assertTrue(p.targets_epoch(2))
        self.assertFalse(p.targets_epoch(3))
        self.assertTrue(p.targets_epoch(4))

    def test_int_epoch_with_stride(self):
        p = NMPulse(config={"epoch": 1, "epoch_delta": 3})
        self.assertFalse(p.targets_epoch(0))
        self.assertTrue(p.targets_epoch(1))
        self.assertFalse(p.targets_epoch(2))
        self.assertFalse(p.targets_epoch(3))
        self.assertTrue(p.targets_epoch(4))
        self.assertTrue(p.targets_epoch(7))


# ---------------------------------------------------------------------------
# NMPulse — waveform shapes
# ---------------------------------------------------------------------------

class TestNMPulseWaveformSquare(unittest.TestCase):

    def test_amp_and_duration_route_from_config(self):
        p = NMPulse(config={"pulse": "square", "amp": 2.0, "onset": 10.0, "duration": 5.0})
        y = p.waveform(100, 0.0, 1.0, 0)
        self.assertAlmostEqual(y[10], 2.0)   # amp applied
        self.assertAlmostEqual(y[14], 2.0)   # inside window
        self.assertAlmostEqual(y[15], 0.0)   # duration truncates


class TestNMPulseWaveformRamp(unittest.TestCase):

    def test_direction_routes_from_config(self):
        p_plus  = NMPulse(config={"pulse": "ramp+", "amp": 1.0, "onset": 0.0, "duration": 10.0})
        p_minus = NMPulse(config={"pulse": "ramp-", "amp": 1.0, "onset": 0.0, "duration": 10.0})
        y_plus  = p_plus.waveform(20, 0.0, 1.0, 0)
        y_minus = p_minus.waveform(20, 0.0, 1.0, 0)
        self.assertAlmostEqual(y_plus[0],  0.0, places=5)   # ramp+ starts at 0
        self.assertAlmostEqual(y_minus[0], 1.0, places=5)   # ramp- starts at amp


class TestNMPulseWaveformExp(unittest.TestCase):

    def test_tau_routes_from_config(self):
        p = NMPulse(config={"pulse": "exp", "amp": 3.0, "onset": 0.0, "tau": 10.0})
        y = p.waveform(100, 0.0, 1.0, 0)
        self.assertAlmostEqual(y[0], 3.0)             # amp at onset
        self.assertAlmostEqual(y[10], 3.0 * math.exp(-1.0), places=5)  # tau decay

    def test_duration_truncates(self):
        p = NMPulse(config={"pulse": "exp", "amp": 1.0, "onset": 0.0, "tau": 10.0, "duration": 20.0})
        y = p.waveform(50, 0.0, 1.0, 0)
        self.assertGreater(y[19], 0.0)
        self.assertAlmostEqual(y[20], 0.0)


class TestNMPulseWaveformAlpha(unittest.TestCase):

    def test_tau_routes_from_config(self):
        p = NMPulse(config={"pulse": "alpha", "amp": 1.0, "onset": 0.0, "tau": 10.0})
        y = p.waveform(100, 0.0, 1.0, 0)
        self.assertAlmostEqual(int(np.argmax(y)), 10, delta=1)  # peak at onset + tau

    def test_duration_truncates(self):
        p = NMPulse(config={"pulse": "alpha", "amp": 1.0, "onset": 0.0, "tau": 10.0, "duration": 15.0})
        y = p.waveform(50, 0.0, 1.0, 0)
        self.assertGreater(y[14], 0.0)
        self.assertAlmostEqual(y[15], 0.0)


# ---------------------------------------------------------------------------
# NMPulse — delta variation
# ---------------------------------------------------------------------------

class TestNMPulseDelta(unittest.TestCase):

    def test_delta_onset_shifts_per_occurrence(self):
        p = NMPulse(config={
            "pulse": "square", "amp": 1.0, "onset": 10.0, "duration": 5.0,
            "onset_delta": 10.0, "epoch_delta": 1,
        })
        # Occurrence 0: onset=10, occurrence 1: onset=20, occurrence 2: onset=30
        y0 = p.waveform(100, 0.0, 1.0, epoch_idx=0)
        y1 = p.waveform(100, 0.0, 1.0, epoch_idx=1)
        y2 = p.waveform(100, 0.0, 1.0, epoch_idx=2)
        # value at onset
        self.assertAlmostEqual(y0[10], 1.0)
        self.assertAlmostEqual(y1[20], 1.0)
        self.assertAlmostEqual(y2[30], 1.0)
        # zero before onset
        self.assertAlmostEqual(y0[9],  0.0)
        self.assertAlmostEqual(y1[19], 0.0)
        self.assertAlmostEqual(y2[29], 0.0)
        # zero after onset + duration
        self.assertAlmostEqual(y0[15], 0.0)
        self.assertAlmostEqual(y1[25], 0.0)
        self.assertAlmostEqual(y2[35], 0.0)

    def test_epoch_delta_stride_occurrence_index(self):
        # epoch=0, epoch_delta=2 → fires at 0,2,4,...
        # occurrence 0 at epoch_idx=0, occurrence 1 at epoch_idx=2
        p = NMPulse(config={
            "pulse": "square", "amp": 1.0, "onset": 10.0, "duration": 5.0,
            "onset_delta": 5.0, "epoch_delta": 2,
        })
        # epoch_idx=0 → occurrence 0 → onset=10
        y0 = p.waveform(100, 0.0, 1.0, epoch_idx=0)
        # epoch_idx=2 → occurrence 1 → onset=15
        y2 = p.waveform(100, 0.0, 1.0, epoch_idx=2)
        self.assertAlmostEqual(y0[10], 1.0)
        self.assertAlmostEqual(y2[15], 1.0)


# ---------------------------------------------------------------------------
# NMPulseContainer
# ---------------------------------------------------------------------------

class TestNMPulseContainerBasic(unittest.TestCase):

    def test_new_returns_pulse(self):
        c = NMPulseContainer()
        p = c.new()
        self.assertIsInstance(p, NMPulse)

    def test_auto_names(self):
        c = NMPulseContainer()
        p0 = c.new()
        p1 = c.new()
        self.assertEqual(p0.name, "p0")
        self.assertEqual(p1.name, "p1")

    def test_len(self):
        c = NMPulseContainer()
        self.assertEqual(len(c), 0)
        c.new()
        self.assertEqual(len(c), 1)
        c.new()
        self.assertEqual(len(c), 2)

    def test_contains(self):
        c = NMPulseContainer()
        c.new()
        self.assertIn("p0", c)
        self.assertNotIn("p1", c)

    def test_getitem(self):
        c = NMPulseContainer()
        p = c.new({"amp": 5.0})
        self.assertIs(c["p0"], p)
        self.assertAlmostEqual(c["p0"].amp, 5.0)

    def test_iter(self):
        c = NMPulseContainer()
        c.new()
        c.new()
        names = [p.name for p in c]
        self.assertEqual(names, ["p0", "p1"])

    def test_new_with_config(self):
        c = NMPulseContainer()
        p = c.new({"pulse": "exp", "tau": 20.0})
        self.assertEqual(p.pulse, "exp")
        self.assertAlmostEqual(p.func.tau, 20.0)


class TestNMPulseContainerRoundTrip(unittest.TestCase):

    def test_to_dict_from_dict(self):
        c = NMPulseContainer()
        c.new({"pulse": "square", "amp": 2.0, "onset": 5.0, "duration": 3.0})
        c.new({"pulse": "exp", "onset": 10.0, "tau": 8.0, "epoch": 1})
        d = c.to_dict()
        c2 = NMPulseContainer.from_dict(d)
        self.assertEqual(len(c2), 2)
        self.assertEqual(c2["p0"].pulse, "square")
        self.assertAlmostEqual(c2["p0"].amp, 2.0)
        self.assertEqual(c2["p1"].pulse, "exp")
        self.assertEqual(c2["p1"].epoch, 1)


# ---------------------------------------------------------------------------
# NMToolPulse — defaults
# ---------------------------------------------------------------------------

class TestNMToolPulseDefaults(unittest.TestCase):

    def setUp(self):
        self.t = NMToolPulse()

    def test_n_points(self):
        self.assertEqual(self.t.n_points, 100)

    def test_xstart(self):
        self.assertAlmostEqual(self.t.xstart, 0.0)

    def test_xdelta(self):
        self.assertAlmostEqual(self.t.xdelta, 1.0)

    def test_prefix(self):
        self.assertEqual(self.t.prefix, "PG_")

    def test_chan(self):
        self.assertEqual(self.t.chan, "")

    def test_overwrite(self):
        self.assertTrue(self.t.overwrite)

    def test_results_to_numpy(self):
        self.assertTrue(self.t.results_to_numpy)

    def test_pulses_empty(self):
        self.assertEqual(len(self.t.pulses), 0)


# ---------------------------------------------------------------------------
# NMToolPulse — property validation
# ---------------------------------------------------------------------------

class TestNMToolPulseProperties(unittest.TestCase):

    def setUp(self):
        self.t = NMToolPulse()

    def test_n_points_valid(self):
        self.t.n_points = 50
        self.assertEqual(self.t.n_points, 50)

    def test_n_points_zero_raises(self):
        with self.assertRaises(ValueError):
            self.t.n_points = 0

    def test_n_points_bool_raises(self):
        with self.assertRaises(TypeError):
            self.t.n_points = True

    def test_n_points_float_raises(self):
        with self.assertRaises(TypeError):
            self.t.n_points = 1.0

    def test_xstart_valid(self):
        self.t.xstart = -5.0
        self.assertAlmostEqual(self.t.xstart, -5.0)

    def test_xstart_bool_raises(self):
        with self.assertRaises(TypeError):
            self.t.xstart = True

    def test_xdelta_valid(self):
        self.t.xdelta = 0.5
        self.assertAlmostEqual(self.t.xdelta, 0.5)

    def test_xdelta_zero_raises(self):
        with self.assertRaises(ValueError):
            self.t.xdelta = 0.0

    def test_xdelta_negative_raises(self):
        with self.assertRaises(ValueError):
            self.t.xdelta = -1.0

    def test_xdelta_bool_raises(self):
        with self.assertRaises(TypeError):
            self.t.xdelta = True

    def test_prefix_valid(self):
        self.t.prefix = "DAC_0_"
        self.assertEqual(self.t.prefix, "DAC_0_")

    def test_prefix_empty_raises(self):
        with self.assertRaises(ValueError):
            self.t.prefix = ""

    def test_prefix_bool_raises(self):
        with self.assertRaises(TypeError):
            self.t.prefix = True

    def test_channel_valid(self):
        self.t.chan = "A"
        self.assertEqual(self.t.chan, "A")

    def test_channel_empty_ok(self):
        self.t.chan = ""
        self.assertEqual(self.t.chan, "")

    def test_channel_bool_raises(self):
        with self.assertRaises(TypeError):
            self.t.chan = True


# ---------------------------------------------------------------------------
# NMToolPulse — pipeline output
# ---------------------------------------------------------------------------

class TestNMToolPulseOutput(unittest.TestCase):

    def setUp(self):
        self.t = NMToolPulse()
        self.t.n_points = 100
        self.t.xstart = 0.0
        self.t.xdelta = 1.0
        self.t.pulses.new({"pulse": "square", "amp": 2.0, "onset": 10.0, "duration": 5.0})

    def test_output_inside_window(self):
        folder = _run_tool(self.t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[10], 2.0)
        self.assertAlmostEqual(y[14], 2.0)

    def test_output_outside_window(self):
        folder = _run_tool(self.t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[9], 0.0)
        self.assertAlmostEqual(y[15], 0.0)

    def test_output_length(self):
        folder = _run_tool(self.t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertEqual(len(y), 100)


# ---------------------------------------------------------------------------
# NMToolPulse — multi-pulse summing
# ---------------------------------------------------------------------------

class TestNMToolPulseMultiPulse(unittest.TestCase):

    def test_overlapping_pulses_sum(self):
        t = NMToolPulse()
        t.n_points = 100
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 10.0, "duration": 5.0})
        t.pulses.new({"pulse": "square", "amp": 3.0, "onset": 10.0, "duration": 5.0})
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[10], 4.0)

    def test_non_overlapping_pulses_dont_sum(self):
        t = NMToolPulse()
        t.n_points = 100
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 5.0, "duration": 5.0})
        t.pulses.new({"pulse": "square", "amp": 2.0, "onset": 20.0, "duration": 5.0})
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[5], 1.0)
        self.assertAlmostEqual(y[20], 2.0)
        self.assertAlmostEqual(y[0], 0.0)


# ---------------------------------------------------------------------------
# NMToolPulse — epoch=0 targeting
# ---------------------------------------------------------------------------

class TestNMToolPulseEpochZeroOnly(unittest.TestCase):

    def test_epoch_zero_pulse_absent_in_later_epochs(self):
        t = NMToolPulse()
        t.n_points = 50
        t.xstart = 0.0
        t.xdelta = 1.0
        # pulse only fires on epoch 0
        t.pulses.new({"pulse": "square", "amp": 5.0, "onset": 5.0, "duration": 5.0,
                      "epoch": 0, "epoch_delta": 100})
        data = [_make_empty_data("w%d" % i) for i in range(3)]
        folder = _run_tool(t, data)
        y0 = folder.data["PG_0"].nparray
        y1 = folder.data["PG_1"].nparray
        y2 = folder.data["PG_2"].nparray
        self.assertAlmostEqual(y0[5], 5.0)
        self.assertAlmostEqual(y1[5], 0.0)
        self.assertAlmostEqual(y2[5], 0.0)


# ---------------------------------------------------------------------------
# NMToolPulse — output arrays (names, xscale, notes)
# ---------------------------------------------------------------------------

class TestNMToolPulseOutputArrays(unittest.TestCase):

    def setUp(self):
        self.t = NMToolPulse()
        self.t.n_points = 50
        self.t.xstart = 2.0
        self.t.xdelta = 0.5
        self.t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 5.0, "duration": 2.0})
        data = [_make_empty_data("rec0")]
        self.folder = _run_tool(self.t, data)

    def test_arrays_written_to_folder(self):
        self.assertIn("PG_0", self.folder.data)

    def test_array_length(self):
        y = self.folder.data["PG_0"].nparray
        self.assertEqual(len(y), 50)

    def test_xscale_start(self):
        d = self.folder.data["PG_0"]
        self.assertAlmostEqual(d.xscale.start, 2.0)

    def test_xscale_delta(self):
        d = self.folder.data["PG_0"]
        self.assertAlmostEqual(d.xscale.delta, 0.5)

    def test_epoch_names_array(self):
        names = self.folder.data["PG_epoch_names"].nparray
        self.assertEqual(list(names), ["rec0"])

    def test_channel_in_array_names(self):
        t = NMToolPulse()
        t.n_points = 20
        t.xstart = 0.0
        t.xdelta = 1.0
        t.chan = "A"
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 2.0, "duration": 2.0})
        folder = _run_tool(t, [_make_empty_data("w0"), _make_empty_data("w1")])
        self.assertIn("PG_A0", folder.data)
        self.assertIn("PG_A1", folder.data)
        self.assertIn("PG_A_epoch_names", folder.data)

    def test_note_contains_pulse_config(self):
        d = self.folder.data["PG_0"]
        notes = getattr(d, "notes", None)
        if notes is None:
            return  # notes not supported on this NMData — skip
        note_text = " ".join(str(n) for n in notes)
        self.assertIn("pulses", note_text)
        # note must include pulse shape
        self.assertIn("square", note_text)


# ---------------------------------------------------------------------------
# NMToolPulse — prefix and channel naming
# ---------------------------------------------------------------------------

class TestNMToolPulsePrefix(unittest.TestCase):

    def _run(self, prefix, channel, n_epochs=2):
        t = NMToolPulse()
        t.n_points = 20
        t.xstart = 0.0
        t.xdelta = 1.0
        t.prefix = prefix
        t.chan = channel
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 2.0, "duration": 2.0})
        data = [_make_empty_data("d%d" % i) for i in range(n_epochs)]
        return _run_tool(t, data)

    def test_default_prefix(self):
        folder = self._run("PG_", "")
        self.assertIn("PG_0", folder.data)
        self.assertIn("PG_1", folder.data)
        self.assertIn("PG_epoch_names", folder.data)

    def test_record_channel(self):
        folder = self._run("Record", "A")
        self.assertIn("RecordA0", folder.data)
        self.assertIn("RecordA1", folder.data)
        self.assertIn("RecordA_epoch_names", folder.data)

    def test_dac_prefix(self):
        folder = self._run("DAC_0_", "")
        self.assertIn("DAC_0_0", folder.data)
        self.assertIn("DAC_0_1", folder.data)
        self.assertIn("DAC_0_epoch_names", folder.data)


# ---------------------------------------------------------------------------
# NMToolPulse — multi-epoch
# ---------------------------------------------------------------------------

class TestNMToolPulseMultiEpoch(unittest.TestCase):

    def test_two_epochs_two_arrays(self):
        t = NMToolPulse()
        t.n_points = 30
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 5.0, "duration": 3.0})
        data = [_make_empty_data("e0"), _make_empty_data("e1")]
        folder = _run_tool(t, data)
        self.assertIn("PG_0", folder.data)
        self.assertIn("PG_1", folder.data)
        self.assertEqual(len(folder.data["PG_epoch_names"].nparray), 2)
        self.assertEqual(list(folder.data["PG_epoch_names"].nparray), ["e0", "e1"])

    def test_delta_onset_across_epochs(self):
        t = NMToolPulse()
        t.n_points = 60
        t.xstart = 0.0
        t.xdelta = 1.0
        # onset shifts by 10 per epoch
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 10.0, "duration": 5.0,
                      "onset_delta": 10.0, "epoch_delta": 1})
        data = [_make_empty_data("e%d" % i) for i in range(3)]
        folder = _run_tool(t, data)
        self.assertAlmostEqual(folder.data["PG_0"].nparray[10], 1.0)
        self.assertAlmostEqual(folder.data["PG_1"].nparray[20], 1.0)
        self.assertAlmostEqual(folder.data["PG_2"].nparray[30], 1.0)


# ---------------------------------------------------------------------------
# NMToolPulseConfig
# ---------------------------------------------------------------------------

class TestNMToolPulseConfig(unittest.TestCase):

    def test_defaults(self):
        c = NMToolPulseConfig()
        self.assertEqual(c.n_points, 100)
        self.assertAlmostEqual(c.xstart, 0.0)
        self.assertAlmostEqual(c.xdelta, 1.0)
        self.assertEqual(c.prefix, "PG_")
        self.assertEqual(c.chan, "")
        self.assertTrue(c.overwrite)

    def test_toml_type(self):
        self.assertEqual(NMToolPulseConfig._TOML_TYPE, "pulse_config")

    def test_set_validates(self):
        c = NMToolPulseConfig()
        with self.assertRaises(ValueError):
            c.n_points = 0
        with self.assertRaises(ValueError):
            c.xdelta = -1.0

    def test_round_trip(self):
        c = NMToolPulseConfig()
        c.n_points = 200
        c.xstart = -5.0
        c.xdelta = 0.5
        c.prefix = "DAC_1_"
        c.chan = "B"
        d = c.to_dict()
        c2 = NMToolPulseConfig.from_dict(d)
        self.assertEqual(c, c2)


# ---------------------------------------------------------------------------
# NMToolPulse — overwrite flag
# ---------------------------------------------------------------------------

class TestNMToolPulseOverwrite(unittest.TestCase):

    def test_overwrite_replaces_arrays(self):
        t = NMToolPulse()
        t.n_points = 20
        t.xstart = 0.0
        t.xdelta = 1.0
        t.overwrite = True
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 2.0, "duration": 2.0})
        folder = NMFolder(NM, name="TestFolder")
        data = [_make_empty_data("w0")]
        targets, _ = _make_targets(data, folder=folder)
        t.run_all(targets)
        count_after_first = len(folder.data)
        t.run_all(targets)  # second run with overwrite=True
        # array count must be identical — arrays replaced, not duplicated
        self.assertEqual(len(folder.data), count_after_first)

    def test_seed_reproduces_gaussian_times(self):
        cfg = {"pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 3.0,
               "n_pulses": 8, "interval": 50.0, "interval_stdv": 10.0,
               "interval_type": "gaussian", "seed": 99}
        folder = NMFolder(NM, name="TestFolder")
        data = [_make_empty_data("w0", n=500)]
        targets, _ = _make_targets(data, folder=folder)
        t = NMToolPulse()
        t.n_points = 500
        t.xstart = 0.0
        t.xdelta = 1.0
        t.overwrite = True
        t.pulses.new(cfg)
        t.run_all(targets)
        times_first = list(folder.toolfolders["Pulse_0"].data["PGT_0"].nparray)
        t.run_all(targets)
        times_second = list(folder.toolfolders["Pulse_0"].data["PGT_0"].nparray)
        self.assertEqual(times_first, times_second)

    def test_overwrite_true_gaussian_times_change(self):
        t = NMToolPulse()
        t.n_points = 500
        t.xstart = 0.0
        t.xdelta = 1.0
        t.overwrite = True
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 3.0,
                      "n_pulses": 5, "interval": 50.0, "interval_stdv": 10.0,
                      "interval_type": "gaussian"})
        folder = NMFolder(NM, name="TestFolder")
        data = [_make_empty_data("w0", n=500)]
        targets, _ = _make_targets(data, folder=folder)
        t.run_all(targets)
        times_first = list(folder.toolfolders["Pulse_0"].data["PGT_0"].nparray)
        t.pulses["p0"].seed = 2
        t.run_all(targets)
        times_second = list(folder.toolfolders["Pulse_0"].data["PGT_0"].nparray)
        self.assertNotEqual(times_first, times_second)

    def test_overwrite_false_raises_on_second_run(self):
        t = NMToolPulse()
        t.n_points = 20
        t.xstart = 0.0
        t.xdelta = 1.0
        t.overwrite = False
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 2.0, "duration": 2.0})
        folder = NMFolder(NM, name="TestFolder")
        data = [_make_empty_data("w0")]
        targets, _ = _make_targets(data, folder=folder)
        t.run_all(targets)
        with self.assertRaises(KeyError):
            t.run_all(targets)  # second run raises — name already exists


# ---------------------------------------------------------------------------
# NMToolPulse — pulse times toolfolder
# ---------------------------------------------------------------------------

class TestNMToolPulseTimes(unittest.TestCase):

    def test_times_toolfolder_created(self):
        t = NMToolPulse()
        t.n_points = 100
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 10.0, "duration": 5.0})
        folder = _run_tool(t, [_make_empty_data("w0")])
        self.assertIn("Pulse_0", folder.toolfolders)

    def test_times_single_pulse_onset(self):
        t = NMToolPulse()
        t.n_points = 100
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 10.0, "duration": 5.0})
        folder = _run_tool(t, [_make_empty_data("w0")])
        times = folder.toolfolders["Pulse_0"].data["PGT_0"].nparray
        self.assertEqual(len(times), 1)
        self.assertAlmostEqual(times[0], 10.0)

    def test_times_fixed_train(self):
        # 3 pulses at onset=0, interval=20 → times [0, 20, 40]
        t = NMToolPulse()
        t.n_points = 100
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 5.0,
                      "n_pulses": 3, "interval": 20.0})
        folder = _run_tool(t, [_make_empty_data("w0")])
        times = folder.toolfolders["Pulse_0"].data["PGT_0"].nparray
        self.assertEqual(len(times), 3)
        self.assertAlmostEqual(times[0], 0.0)
        self.assertAlmostEqual(times[1], 20.0)
        self.assertAlmostEqual(times[2], 40.0)

    def test_times_multiple_epochs(self):
        t = NMToolPulse()
        t.n_points = 100
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 5.0, "duration": 3.0,
                      "epoch": "all"})
        folder = _run_tool(t, [_make_empty_data("w0"), _make_empty_data("w1")])
        tf = folder.toolfolders["Pulse_0"]
        self.assertIn("PGT_0", tf.data)
        self.assertIn("PGT_1", tf.data)

    def test_times_sorted_across_pulses(self):
        # two pulses with different onsets — times should be sorted
        t = NMToolPulse()
        t.n_points = 100
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 30.0, "duration": 3.0})
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 10.0, "duration": 3.0})
        folder = _run_tool(t, [_make_empty_data("w0")])
        times = folder.toolfolders["Pulse_0"].data["PGT_0"].nparray
        self.assertEqual(list(times), sorted(times))

    def test_times_channel_prefix(self):
        t = NMToolPulse()
        t.n_points = 50
        t.xstart = 0.0
        t.xdelta = 1.0
        t.chan = "A"
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 5.0, "duration": 3.0})
        folder = _run_tool(t, [_make_empty_data("w0")])
        self.assertIn("PGT_A0", folder.toolfolders["Pulse_0"].data)


# ---------------------------------------------------------------------------
# NMToolPulse — enabled flag
# ---------------------------------------------------------------------------

class TestNMToolPulseEnabled(unittest.TestCase):

    def test_disabled_pulse_produces_zeros(self):
        t = NMToolPulse()
        t.n_points = 50
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "square", "amp": 2.0, "onset": 10.0, "duration": 5.0,
                      "enabled": False})
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        np.testing.assert_array_equal(y, 0.0)

    def test_disabled_pulse_skipped_other_active(self):
        t = NMToolPulse()
        t.n_points = 50
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 5.0, "duration": 5.0})
        t.pulses.new({"pulse": "square", "amp": 3.0, "onset": 5.0, "duration": 5.0,
                      "enabled": False})
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[5], 1.0)   # only first pulse contributes

    def test_round_trip_enabled_false(self):
        p = NMPulse(config={"pulse": "square", "amp": 1.0, "onset": 5.0,
                             "enabled": False})
        self.assertFalse(p.enabled)
        p2 = NMPulse.from_dict(p.to_dict())
        self.assertFalse(p2.enabled)


# ---------------------------------------------------------------------------
# NMPulse — sin / cos / sinzap / user waveforms
# ---------------------------------------------------------------------------

class TestNMPulseWaveformSin(unittest.TestCase):

    def test_freq_phase_route_from_config(self):
        p = NMPulse(config={"pulse": "sin", "freq": 0.25, "phase": math.pi / 2,
                             "amp": 1.0, "onset": 0.0})
        y = p.waveform(20, 0.0, 1.0, 0)
        self.assertAlmostEqual(y[0], 1.0, places=5)   # phase=π/2 → cos(0)=1

    def test_duration_truncates(self):
        p = NMPulse(config={"pulse": "sin", "freq": 0.25,
                             "amp": 1.0, "onset": 5.0, "duration": 4.0})
        y = p.waveform(30, 0.0, 1.0, 0)
        self.assertAlmostEqual(y[9], 0.0)

    def test_func_property_accessible(self):
        p = NMPulse(config={"pulse": "sin"})
        from pyneuromatic.analysis.nm_pulse_func import NMPulseFuncSin
        self.assertIsInstance(p.func, NMPulseFuncSin)

    def test_round_trip_dict(self):
        p = NMPulse(config={"pulse": "sin", "freq": 0.5, "phase": 0.1,
                             "amp": 1.0, "onset": 0.0})
        d = p.to_dict()
        self.assertEqual(d["pulse"], "sin")
        self.assertAlmostEqual(d["freq"], 0.5)
        self.assertAlmostEqual(d["phase"], 0.1)
        p2 = NMPulse(config=d)
        self.assertAlmostEqual(p2.func._freq, 0.5)
        self.assertAlmostEqual(p2.func._phase, 0.1)


class TestNMPulseWaveformSinZap(unittest.TestCase):

    def test_truncated_by_duration(self):
        p = NMPulse(config={"pulse": "sinzap", "f0": 0.05, "f1": 0.2,
                             "amp": 1.0, "onset": 0.0, "duration": 50.0})
        y = p.waveform(100, 0.0, 1.0, 0)
        self.assertAlmostEqual(y[50], 0.0)

    def test_round_trip_dict(self):
        p = NMPulse(config={"pulse": "sinzap", "f0": 0.1, "f1": 0.5,
                             "amp": 1.0, "onset": 0.0})
        d = p.to_dict()
        self.assertEqual(d["pulse"], "sinzap")
        self.assertAlmostEqual(d["f0"], 0.1)
        self.assertAlmostEqual(d["f1"], 0.5)
        p2 = NMPulse(config=d)
        self.assertAlmostEqual(p2.func._f0, 0.1)
        self.assertAlmostEqual(p2.func._f1, 0.5)


_USER_RAW = np.sin(np.linspace(0, 2 * math.pi, 50))


class TestNMPulseWaveformUser(unittest.TestCase):

    def _make_user_pulse(self, onset=0.0, amp=1.0, duration=math.inf):
        return NMPulse(config={"pulse": "user", "amp": amp, "onset": onset,
                                "duration": duration, "data": _USER_RAW,
                                "data_xdelta": 1.0})

    def test_onset_shifts_waveform(self):
        p0 = self._make_user_pulse(onset=0.0, amp=1.0)
        p10 = self._make_user_pulse(onset=10.0, amp=1.0)
        y0 = p0.waveform(100, 0.0, 1.0, 0)
        y10 = p10.waveform(100, 0.0, 1.0, 0)
        # peak of shifted wave is 10 samples later
        self.assertAlmostEqual(np.argmax(y0) + 10, np.argmax(y10), delta=1)

    def test_func_property_accessible(self):
        p = self._make_user_pulse()
        from pyneuromatic.analysis.nm_pulse_func import NMPulseFuncUser
        self.assertIsInstance(p.func, NMPulseFuncUser)


# ---------------------------------------------------------------------------
# NMToolPulse — sin / cos / sinzap / user shapes via run_all
# ---------------------------------------------------------------------------

class TestNMToolPulseSin(unittest.TestCase):

    def setUp(self):
        self.t = NMToolPulse()
        self.t.n_points = 100
        self.t.xstart = 0.0
        self.t.xdelta = 1.0
        self.t.pulses.new({"pulse": "sin", "freq": 0.25, "phase": 0.0,
                           "amp": 2.0, "onset": 0.0})

    def test_peak_at_quarter_period(self):
        folder = _run_tool(self.t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[1], 2.0, places=5)

    def test_output_length(self):
        folder = _run_tool(self.t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertEqual(len(y), 100)

    def test_zero_before_onset(self):
        t = NMToolPulse()
        t.n_points = 100
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "sin", "freq": 0.1, "amp": 1.0, "onset": 20.0})
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[19], 0.0)


class TestNMToolPulseSinZap(unittest.TestCase):

    def setUp(self):
        self.t = NMToolPulse()
        self.t.n_points = 200
        self.t.xstart = 0.0
        self.t.xdelta = 1.0
        self.t.pulses.new({"pulse": "sinzap", "f0": 0.05, "f1": 0.2,
                           "amp": 1.0, "onset": 0.0})

    def test_zero_at_onset(self):
        folder = _run_tool(self.t, [_make_empty_data("w0", n=200)])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[0], 0.0, places=10)

    def test_nonzero_in_sweep(self):
        folder = _run_tool(self.t, [_make_empty_data("w0", n=200)])
        y = folder.data["PG_0"].nparray
        self.assertGreater(np.max(np.abs(y)), 0.5)

    def test_output_length(self):
        folder = _run_tool(self.t, [_make_empty_data("w0", n=200)])
        y = folder.data["PG_0"].nparray
        self.assertEqual(len(y), 200)

    def test_truncated_by_duration(self):
        t = NMToolPulse()
        t.n_points = 200
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "sinzap", "f0": 0.05, "f1": 0.2,
                      "amp": 1.0, "onset": 0.0, "duration": 50.0})
        folder = _run_tool(t, [_make_empty_data("w0", n=200)])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[50], 0.0)


class TestNMToolPulseUser(unittest.TestCase):

    def _make_tool(self, onset=0.0, amp=1.0, duration=math.inf, n=100):
        t = NMToolPulse()
        t.n_points = n
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "user", "amp": amp, "onset": onset,
                      "duration": duration, "data": _USER_RAW, "data_xdelta": 1.0})
        return t

    def test_peak_equals_amp(self):
        t = self._make_tool(amp=2.0)
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(np.max(y), 2.0, places=4)

    def test_zero_before_onset(self):
        t = self._make_tool(onset=20.0)
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[19], 0.0)

    def test_truncation_by_duration(self):
        t = self._make_tool(onset=0.0, duration=20.0)
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[20], 0.0)

    def test_output_length(self):
        t = self._make_tool()
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertEqual(len(y), 100)


# ---------------------------------------------------------------------------
# NMPulse — train defaults and properties
# ---------------------------------------------------------------------------

class TestNMPulseTrainDefaults(unittest.TestCase):

    def setUp(self):
        self.p = NMPulse()

    def test_n_pulses(self):
        self.assertEqual(self.p.n_pulses, 1)

    def test_interval(self):
        self.assertAlmostEqual(self.p.interval, 100.0)

    def test_interval_stdv(self):
        self.assertAlmostEqual(self.p.interval_stdv, 0.0)

    def test_interval_type(self):
        self.assertEqual(self.p.interval_type, "fixed")

    def test_train_duration(self):
        self.assertTrue(math.isinf(self.p.train_duration))


class TestNMPulseTrainProperties(unittest.TestCase):

    def setUp(self):
        self.p = NMPulse()

    def test_n_pulses_valid(self):
        self.p.n_pulses = 5
        self.assertEqual(self.p.n_pulses, 5)

    def test_n_pulses_zero_allowed(self):
        self.p.n_pulses = 0
        self.assertEqual(self.p.n_pulses, 0)

    def test_n_pulses_negative_raises(self):
        with self.assertRaises(ValueError):
            self.p.n_pulses = -1

    def test_n_pulses_bool_raises(self):
        with self.assertRaises(TypeError):
            self.p.n_pulses = True

    def test_interval_valid(self):
        self.p.interval = 50.0
        self.assertAlmostEqual(self.p.interval, 50.0)

    def test_interval_zero_raises(self):
        with self.assertRaises(ValueError):
            self.p.interval = 0.0

    def test_interval_negative_raises(self):
        with self.assertRaises(ValueError):
            self.p.interval = -10.0

    def test_interval_stdv_valid(self):
        self.p.interval_stdv = 5.0
        self.assertAlmostEqual(self.p.interval_stdv, 5.0)

    def test_interval_stdv_zero_allowed(self):
        self.p.interval_stdv = 0.0
        self.assertAlmostEqual(self.p.interval_stdv, 0.0)

    def test_interval_stdv_negative_raises(self):
        with self.assertRaises(ValueError):
            self.p.interval_stdv = -1.0

    def test_interval_type_valid(self):
        for t in ("fixed", "gaussian", "poisson"):
            self.p.interval_type = t
            self.assertEqual(self.p.interval_type, t)

    def test_interval_type_invalid_raises(self):
        with self.assertRaises(ValueError):
            self.p.interval_type = "random"

    def test_interval_type_bool_raises(self):
        with self.assertRaises(TypeError):
            self.p.interval_type = True

    def test_interval_min_valid(self):
        self.p.interval_min = 5.0
        self.assertAlmostEqual(self.p.interval_min, 5.0)

    def test_interval_min_zero_allowed(self):
        self.p.interval_min = 0.0
        self.assertAlmostEqual(self.p.interval_min, 0.0)

    def test_interval_min_negative_raises(self):
        with self.assertRaises(ValueError):
            self.p.interval_min = -1.0

    def test_interval_max_valid(self):
        self.p.interval_max = 200.0
        self.assertAlmostEqual(self.p.interval_max, 200.0)

    def test_interval_max_zero_raises(self):
        with self.assertRaises(ValueError):
            self.p.interval_max = 0.0

    def test_train_duration_valid(self):
        self.p.train_duration = 500.0
        self.assertAlmostEqual(self.p.train_duration, 500.0)

    def test_train_duration_zero_raises(self):
        with self.assertRaises(ValueError):
            self.p.train_duration = 0.0

    def test_train_duration_negative_raises(self):
        with self.assertRaises(ValueError):
            self.p.train_duration = -1.0


# ---------------------------------------------------------------------------
# NMPulse — train waveform
# ---------------------------------------------------------------------------

class TestNMPulseTrainWaveform(unittest.TestCase):

    def test_single_pulse_n_pulses_one(self):
        # n_pulses=1 (default) produces same result as without train params
        p1 = NMPulse(config={"pulse": "square", "amp": 2.0, "onset": 10.0, "duration": 5.0})
        p2 = NMPulse(config={"pulse": "square", "amp": 2.0, "onset": 10.0, "duration": 5.0,
                              "n_pulses": 1, "interval": 100.0})
        y1 = p1.waveform(50, 0.0, 1.0, 0)
        y2 = p2.waveform(50, 0.0, 1.0, 0)
        np.testing.assert_array_almost_equal(y1, y2)

    def test_fixed_train_pulse_positions(self):
        # 3 pulses: onset=0, interval=20 → pulses at 0, 20, 40
        p = NMPulse(config={
            "pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 5.0,
            "n_pulses": 3, "interval": 20.0,
        })
        y = p.waveform(100, 0.0, 1.0, 0)
        for start in (0, 20, 40):
            self.assertAlmostEqual(y[start], 1.0)
            self.assertAlmostEqual(y[start + 4], 1.0)
            self.assertAlmostEqual(y[start + 5], 0.0)

    def test_pulses_sum_when_overlapping(self):
        p = NMPulse(config={
            "pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 15.0,
            "n_pulses": 2, "interval": 10.0,
        })
        y = p.waveform(50, 0.0, 1.0, 0)
        self.assertAlmostEqual(y[10], 2.0)

    def test_output_length(self):
        p = NMPulse(config={
            "pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 3.0,
            "n_pulses": 4, "interval": 10.0,
        })
        y = p.waveform(80, 0.0, 1.0, 0)
        self.assertEqual(len(y), 80)

    def test_gaussian_interval_differs_from_fixed(self):
        cfg = {"pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 5.0,
               "n_pulses": 5, "interval": 20.0}
        p1 = NMPulse(config={**cfg, "interval_type": "fixed"})
        p2 = NMPulse(config={**cfg, "interval_type": "gaussian",
                              "interval_stdv": 5.0, "seed": 42})
        y_fixed    = p1.waveform(200, 0.0, 1.0, 0)
        y_gaussian = p2.waveform(200, 0.0, 1.0, 0)
        self.assertFalse(np.array_equal(y_fixed, y_gaussian))

    def test_interval_min_enforced(self):
        # interval_min=15 on a gaussian train with mean=10, large stdv
        # all intervals should be >= 15
        p = NMPulse(config={
            "pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 3.0,
            "n_pulses": 10, "interval": 10.0, "interval_stdv": 8.0,
            "interval_type": "gaussian", "interval_min": 15.0, "seed": 42,
        })
        p.waveform(200, 0.0, 1.0, 0)
        times = p._last_onset_times
        for i in range(1, len(times)):
            self.assertGreaterEqual(times[i] - times[i - 1], 15.0 - 1e-10)

    def test_interval_max_enforced(self):
        # interval_max=25 on a poisson train with mean=100 (very long gaps)
        # all intervals should be <= 25
        p = NMPulse(config={
            "pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 3.0,
            "n_pulses": 5, "interval": 100.0,
            "interval_type": "poisson", "interval_max": 25.0, "seed": 42,
        })
        p.waveform(200, 0.0, 1.0, 0)
        times = p._last_onset_times
        for i in range(1, len(times)):
            self.assertLessEqual(times[i] - times[i - 1], 25.0)

    def test_train_duration_limits_pulses(self):
        # interval=20, train_duration=50 → pulses at 0, 20, 40; onset=60 excluded
        p = NMPulse(config={
            "pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 5.0,
            "n_pulses": 0, "interval": 20.0, "train_duration": 50.0,
        })
        y = p.waveform(100, 0.0, 1.0, 0)
        for start in (0, 20, 40):
            self.assertAlmostEqual(y[start], 1.0)
        self.assertAlmostEqual(y[60], 0.0)   # pulse at onset=60 excluded
        self.assertAlmostEqual(y[80], 0.0)   # pulse at onset=80 excluded

    def test_n_pulses_and_train_duration_both_cap(self):
        # n_pulses=2 wins over train_duration=100 (only 2 pulses generated)
        p = NMPulse(config={
            "pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 5.0,
            "n_pulses": 2, "interval": 20.0, "train_duration": 100.0,
        })
        y = p.waveform(150, 0.0, 1.0, 0)
        self.assertAlmostEqual(y[0],  1.0)
        self.assertAlmostEqual(y[20], 1.0)
        self.assertAlmostEqual(y[40], 0.0)   # third pulse not generated
        self.assertAlmostEqual(y[60], 0.0)   # fourth pulse not generated
        self.assertAlmostEqual(y[80], 0.0)   # fifth pulse not generated

    def test_to_dict_round_trip(self):
        p = NMPulse(config={
            "pulse": "exp", "tau": 5.0, "amp": 2.0, "onset": 10.0,
            "n_pulses": 0, "interval": 30.0, "train_duration": 200.0,
            "interval_type": "gaussian", "interval_stdv": 1.0,
        })
        d = p.to_dict()
        self.assertEqual(d["n_pulses"], 0)
        self.assertAlmostEqual(d["train_duration"], 200.0)
        self.assertEqual(d["interval_type"], "gaussian")
        p2 = NMPulse.from_dict(d)
        self.assertEqual(p, p2)


# ---------------------------------------------------------------------------
# NMToolPulse — Poisson train via PGT_ times
# ---------------------------------------------------------------------------

class TestNMToolPulsePoisson(unittest.TestCase):

    def _run_poisson(self, n_pulses, mean_interval, seed=0):
        t = NMToolPulse()
        t.n_points = 5000
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 3.0,
                      "n_pulses": n_pulses, "interval": mean_interval,
                      "interval_type": "poisson", "seed": seed})
        folder = _run_tool(t, [_make_empty_data("w0", n=5000)])
        return folder.toolfolders["Pulse_0"].data["PGT_0"].nparray

    def test_correct_number_of_pulses(self):
        times = self._run_poisson(n_pulses=10, mean_interval=50.0)
        self.assertEqual(len(times), 10)

    def test_all_intervals_positive(self):
        times = self._run_poisson(n_pulses=20, mean_interval=50.0)
        for i in range(1, len(times)):
            self.assertGreater(times[i] - times[i - 1], 0.0)

    def test_intervals_differ_from_fixed(self):
        # Poisson intervals should not all equal the mean interval
        times = self._run_poisson(n_pulses=10, mean_interval=50.0, seed=42)
        intervals = [times[i] - times[i - 1] for i in range(1, len(times))]
        self.assertFalse(all(abs(iv - 50.0) < 1e-10 for iv in intervals))

    def test_times_sorted(self):
        times = self._run_poisson(n_pulses=15, mean_interval=30.0)
        self.assertEqual(list(times), sorted(times))


# ---------------------------------------------------------------------------
# NMPulseContainer — train via n_pulses
# ---------------------------------------------------------------------------

class TestNMPulseContainerTrain(unittest.TestCase):

    def test_new_train_returns_nmPulse(self):
        c = NMPulseContainer()
        p = c.new({"pulse": "square", "amp": 1.0, "onset": 0.0, "duration": 5.0,
                   "n_pulses": 3, "interval": 20.0})
        self.assertIsInstance(p, NMPulse)
        self.assertEqual(p.n_pulses, 3)

    def test_container_round_trip(self):
        c = NMPulseContainer()
        c.new({"pulse": "exp", "amp": 1.0, "onset": 5.0, "tau": 10.0})
        c.new({"pulse": "square", "amp": 2.0, "onset": 0.0, "duration": 5.0,
               "n_pulses": 3, "interval": 20.0})
        d = c.to_dict()
        c2 = NMPulseContainer.from_dict(d)
        self.assertIsInstance(c2["p0"], NMPulse)
        self.assertIsInstance(c2["p1"], NMPulse)
        self.assertEqual(c2["p0"].n_pulses, 1)
        self.assertEqual(c2["p1"].n_pulses, 3)


# ---------------------------------------------------------------------------
# NMToolPulse — train via run_all
# ---------------------------------------------------------------------------

class TestNMToolPulseTrain(unittest.TestCase):

    def test_fixed_train_output(self):
        t = NMToolPulse()
        t.n_points = 100
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"type": "train", "pulse": "square",
                      "amp": 1.0, "onset": 0.0, "duration": 5.0,
                      "n_pulses": 3, "interval": 20.0})
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertEqual(len(y), 100)
        for start in (0, 20, 40):
            self.assertAlmostEqual(y[start], 1.0)
            self.assertAlmostEqual(y[start + 5], 0.0)

    def test_train_note_contains_n_pulses(self):
        t = NMToolPulse()
        t.n_points = 100
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"type": "train", "pulse": "square",
                      "amp": 1.0, "onset": 0.0, "duration": 5.0,
                      "n_pulses": 4, "interval": 20.0})
        folder = _run_tool(t, [_make_empty_data("w0")])
        d = folder.data["PG_0"]
        notes = getattr(d, "notes", None)
        if notes is None:
            return
        note_text = " ".join(str(n) for n in notes)
        self.assertIn("n_pulses=4", note_text)
        self.assertIn("interval=20", note_text)


if __name__ == "__main__":
    unittest.main()
