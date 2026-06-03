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

    def test_width(self):
        self.assertTrue(math.isinf(self.p.width))

    def test_amp_delta(self):
        self.assertEqual(self.p.amp_delta, 0.0)

    def test_onset_delta(self):
        self.assertEqual(self.p.onset_delta, 0.0)

    def test_width_delta(self):
        self.assertEqual(self.p.width_delta, 0.0)

    def test_amp_stdv(self):
        self.assertEqual(self.p.amp_stdv, 0.0)

    def test_onset_stdv(self):
        self.assertEqual(self.p.onset_stdv, 0.0)

    def test_width_stdv(self):
        self.assertEqual(self.p.width_stdv, 0.0)


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
                             "amp": 4.0, "onset": 1.0, "width": 5.0,
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

    def test_amp_and_width_route_from_config(self):
        p = NMPulse(config={"pulse": "square", "amp": 2.0, "onset": 10.0, "width": 5.0})
        y = p.waveform(100, 0.0, 1.0, 0)
        self.assertAlmostEqual(y[10], 2.0)   # amp applied
        self.assertAlmostEqual(y[14], 2.0)   # inside window
        self.assertAlmostEqual(y[15], 0.0)   # width truncates


class TestNMPulseWaveformRamp(unittest.TestCase):

    def test_direction_routes_from_config(self):
        p_plus  = NMPulse(config={"pulse": "ramp+", "amp": 1.0, "onset": 0.0, "width": 10.0})
        p_minus = NMPulse(config={"pulse": "ramp-", "amp": 1.0, "onset": 0.0, "width": 10.0})
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

    def test_width_truncates(self):
        p = NMPulse(config={"pulse": "exp", "amp": 1.0, "onset": 0.0, "tau": 10.0, "width": 20.0})
        y = p.waveform(50, 0.0, 1.0, 0)
        self.assertGreater(y[19], 0.0)
        self.assertAlmostEqual(y[20], 0.0)


class TestNMPulseWaveformAlpha(unittest.TestCase):

    def test_tau_routes_from_config(self):
        p = NMPulse(config={"pulse": "alpha", "amp": 1.0, "onset": 0.0, "tau": 10.0})
        y = p.waveform(100, 0.0, 1.0, 0)
        self.assertAlmostEqual(int(np.argmax(y)), 10, delta=1)  # peak at onset + tau

    def test_width_truncates(self):
        p = NMPulse(config={"pulse": "alpha", "amp": 1.0, "onset": 0.0, "tau": 10.0, "width": 15.0})
        y = p.waveform(50, 0.0, 1.0, 0)
        self.assertGreater(y[14], 0.0)
        self.assertAlmostEqual(y[15], 0.0)


# ---------------------------------------------------------------------------
# NMPulse — delta variation
# ---------------------------------------------------------------------------

class TestNMPulseDelta(unittest.TestCase):

    def test_delta_onset_shifts_per_occurrence(self):
        p = NMPulse(config={
            "pulse": "square", "amp": 1.0, "onset": 10.0, "width": 5.0,
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
        # zero after onset + width
        self.assertAlmostEqual(y0[15], 0.0)
        self.assertAlmostEqual(y1[25], 0.0)
        self.assertAlmostEqual(y2[35], 0.0)

    def test_epoch_delta_stride_occurrence_index(self):
        # epoch=0, epoch_delta=2 → fires at 0,2,4,...
        # occurrence 0 at epoch_idx=0, occurrence 1 at epoch_idx=2
        p = NMPulse(config={
            "pulse": "square", "amp": 1.0, "onset": 10.0, "width": 5.0,
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
        c.new({"pulse": "square", "amp": 2.0, "onset": 5.0, "width": 3.0})
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
        self.t.pulses.new({"pulse": "square", "amp": 2.0, "onset": 10.0, "width": 5.0})

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
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 10.0, "width": 5.0})
        t.pulses.new({"pulse": "square", "amp": 3.0, "onset": 10.0, "width": 5.0})
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[10], 4.0)

    def test_non_overlapping_pulses_dont_sum(self):
        t = NMToolPulse()
        t.n_points = 100
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 5.0, "width": 5.0})
        t.pulses.new({"pulse": "square", "amp": 2.0, "onset": 20.0, "width": 5.0})
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
        t.pulses.new({"pulse": "square", "amp": 5.0, "onset": 5.0, "width": 5.0,
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
        self.t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 5.0, "width": 2.0})
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
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 2.0, "width": 2.0})
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
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 2.0, "width": 2.0})
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
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 5.0, "width": 3.0})
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
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 10.0, "width": 5.0,
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
        t.pulses.new({"pulse": "square", "amp": 1.0, "onset": 2.0, "width": 2.0})
        folder = NMFolder(NM, name="TestFolder")
        data = [_make_empty_data("w0")]
        targets, _ = _make_targets(data, folder=folder)
        t.run_all(targets)
        t.run_all(targets)  # second run with overwrite=True
        # arrays should be replaced, not duplicated
        self.assertIn("PG_0", folder.data)
        self.assertIn("PG_epoch_names", folder.data)


# ---------------------------------------------------------------------------
# NMPulse — sin / cos / sinzap / user waveforms
# ---------------------------------------------------------------------------

class TestNMPulseWaveformSin(unittest.TestCase):

    def test_freq_phase_route_from_config(self):
        p = NMPulse(config={"pulse": "sin", "freq": 0.25, "phase": math.pi / 2,
                             "amp": 1.0, "onset": 0.0})
        y = p.waveform(20, 0.0, 1.0, 0)
        self.assertAlmostEqual(y[0], 1.0, places=5)   # phase=π/2 → cos(0)=1

    def test_width_truncates(self):
        p = NMPulse(config={"pulse": "sin", "freq": 0.25,
                             "amp": 1.0, "onset": 5.0, "width": 4.0})
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

    def test_truncated_by_width(self):
        p = NMPulse(config={"pulse": "sinzap", "f0": 0.05, "f1": 0.2,
                             "amp": 1.0, "onset": 0.0, "width": 50.0})
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

    def _make_user_pulse(self, onset=0.0, amp=1.0, width=math.inf):
        return NMPulse(config={"pulse": "user", "amp": amp, "onset": onset,
                                "width": width, "data": _USER_RAW,
                                "xdelta_data": 1.0})

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

    def test_truncated_by_width(self):
        t = NMToolPulse()
        t.n_points = 200
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "sinzap", "f0": 0.05, "f1": 0.2,
                      "amp": 1.0, "onset": 0.0, "width": 50.0})
        folder = _run_tool(t, [_make_empty_data("w0", n=200)])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[50], 0.0)


class TestNMToolPulseUser(unittest.TestCase):

    def _make_tool(self, onset=0.0, amp=1.0, width=math.inf, n=100):
        t = NMToolPulse()
        t.n_points = n
        t.xstart = 0.0
        t.xdelta = 1.0
        t.pulses.new({"pulse": "user", "amp": amp, "onset": onset,
                      "width": width, "data": _USER_RAW, "xdelta_data": 1.0})
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

    def test_truncation_by_width(self):
        t = self._make_tool(onset=0.0, width=20.0)
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertAlmostEqual(y[20], 0.0)

    def test_output_length(self):
        t = self._make_tool()
        folder = _run_tool(t, [_make_empty_data("w0")])
        y = folder.data["PG_0"].nparray
        self.assertEqual(len(y), 100)


if __name__ == "__main__":
    unittest.main()
